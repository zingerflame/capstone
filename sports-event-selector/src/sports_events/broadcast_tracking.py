"""Local camera-motion segmentation, detection, tracking and cross-shot candidates.

Run on original video BEFORE delta filtering. Similar appearance is not proof of
identity. Named identities require reviewed bindings; uncertain objects stay anonymous.
"""
import argparse
from collections import deque, Counter
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import time
import cv2
import numpy as np

from .frame_identity import video_frames, frame_key
from .io import read_json, write_json
from .vertex_dataset import sha

YOLOX_SHA = 'c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063'
YOLOX_URL = 'https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/object_detection_yolox/object_detection_yolox_2022nov.onnx'
OSNET_SHA = 'cf55163d78fc44c62c82f85ab62d39f10438679b5abe8c698ae08cfa84aa6e18'


@dataclass
class TrackingConfig:
    width: int = 640
    max_step_pixels: float = .7
    max_drift_pixels: float = 2.
    min_stable_seconds: float = .6
    detect_every: int = 5
    confidence: float = .30
    max_track_gap_seconds: float = .6
    association_threshold: float = .78
    association_margin: float = .08
    scoreboard: tuple = (215/1280, 38/720, 423/1280, 100/720)

    def __post_init__(self):
        if self.width<160 or self.detect_every<1 or not 0<self.confidence<=1:
            raise ValueError('Invalid detection size, interval or confidence')
        if min(self.max_step_pixels,self.max_drift_pixels,self.min_stable_seconds,self.max_track_gap_seconds)<=0:
            raise ValueError('Motion, duration and gap limits must be positive')
        if not 0<=self.association_threshold<=1 or not 0<=self.association_margin<=1:
            raise ValueError('Invalid appearance matching limits')


def pixel_box(box, width, height):
    if len(box) != 4 or not all(np.isfinite(v) for v in box) or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
        raise ValueError('Expected normalized x1,y1,x2,y2 box')
    return [int(box[0]*width), int(box[1]*height), min(width, int(np.ceil(box[2]*width))), min(height, int(np.ceil(box[3]*height)))]


def camera_step(previous, current, excluded=()):
    """RANSAC background affine motion; poor texture/coverage is unknown, not stable."""
    h,w = current.shape; mask = np.full((h,w),255,np.uint8)
    for box in excluded:
        x1,y1,x2,y2 = pixel_box(box,w,h); mask[y1:y2,x1:x2] = 0
    points = cv2.goodFeaturesToTrack(previous,500,.015,8,mask=mask)
    if points is None or len(points)<20: return {'status':'unknown','reason':'too_few_features'}
    found,good,_ = cv2.calcOpticalFlowPyrLK(previous,current,points,None,winSize=(21,21),maxLevel=3)
    back,back_good,_ = cv2.calcOpticalFlowPyrLK(current,previous,found,None,winSize=(21,21),maxLevel=3)
    keep = good.ravel().astype(bool) & back_good.ravel().astype(bool) & (np.linalg.norm(back-points,axis=2).ravel()<1.5)
    a,b = points[keep].reshape(-1,2),found[keep].reshape(-1,2)
    if len(a)<20: return {'status':'unknown','reason':'lost_features'}
    transform,inliers = cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=1.5)
    if transform is None: return {'status':'unknown','reason':'affine_failed'}
    good = inliers.ravel().astype(bool)
    cells = {(min(3,int(x/w*4)),min(2,int(y/h*3))) for x,y in a[good]}
    if good.mean()<.55 or len(cells)<4: return {'status':'unknown','reason':'insufficient_background_consensus'}
    anchors = np.array([[0,0],[w,0],[w,h],[0,h],[w/2,h/2]],np.float32)
    moved = cv2.transform(anchors[None],transform)[0]
    displacement = float(np.max(np.linalg.norm(moved-anchors,axis=1)))
    return {'status':'estimated','step_pixels':displacement,'inlier_fraction':float(good.mean()),
            'background_cells':len(cells),'affine':transform.tolist()}


def stable_segments(rows, fps, minimum_seconds):
    result=[]; start=None
    for i in range(len(rows)+1):
        stable = i<len(rows) and rows[i]['camera']['status']=='stable'
        if stable and start is None: start=i
        if not stable and start is not None:
            if i-start >= max(2,int(np.ceil(fps*minimum_seconds))):
                sid=f'shot_{len(result):04d}'
                result.append({'id':sid,'start_frame':rows[start]['source_frame'],
                               'end_frame_exclusive':rows[i-1]['source_frame']+1})
                for row in rows[start:i]: row['segment_id']=sid
            start=None
    return result


def scan(video, start, end, config):
    rows=[]; previous=None; transforms=deque(); fps=None
    for meta,image in video_frames(video,start,end):
        fps=meta['source_fps']; h,w=image.shape[:2]
        gray=cv2.cvtColor(cv2.resize(image,(config.width,round(h*config.width/w))),cv2.COLOR_BGR2GRAY)
        if previous is None: motion={'status':'unknown','reason':'first_frame'}
        else:
            motion=camera_step(previous,gray,[config.scoreboard])
            if motion['status']=='estimated':
                matrix=np.vstack([motion['affine'],[0,0,1]])
                transforms.append(matrix)
                while len(transforms)>max(2,round(fps*.5)): transforms.popleft()
                combined=np.eye(3)
                for t in transforms: combined=t@combined
                corners=np.array([[0,0,1],[gray.shape[1],0,1],[0,gray.shape[0],1],[gray.shape[1],gray.shape[0],1]])
                drift=float(np.max(np.linalg.norm((corners@combined.T-corners)[:,:2],axis=1)))
                motion['drift_pixels']=drift
                motion['status']='stable' if motion['step_pixels']<=config.max_step_pixels and drift<=config.max_drift_pixels else 'moving'
            else: transforms.clear()
        rows.append({**meta,'segment_id':None,'camera':motion});previous=gray
        if len(rows)%250==0: print('camera frames',len(rows),flush=True)
    if not rows: raise ValueError('No source frames in requested interval')
    segments=stable_segments(rows,fps,config.min_stable_seconds)
    return rows,segments


class Detector:
    """YOLOX-S COCO ONNX; independent decoding using the published YOLOX equations."""
    def __init__(self, weights, confidence=.3):
        if sha(weights)!=YOLOX_SHA: raise ValueError('Unexpected YOLOX weight checksum')
        self.net=cv2.dnn.readNet(str(weights));self.confidence=confidence
        anchors=[];scales=[]
        for stride in (8,16,32):
            y,x=np.mgrid[:640//stride,:640//stride]
            anchors.append(np.column_stack((x.ravel(),y.ravel())))
            scales.extend([stride]*x.size)
        self.anchors=np.concatenate(anchors);self.scales=np.array(scales)[:,None]

    def __call__(self,image):
        h,w=image.shape[:2];ratio=min(640/w,640/h)
        canvas=np.full((640,640,3),114,np.float32)
        resized=cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),(int(w*ratio),int(h*ratio)))
        canvas[:resized.shape[0],:resized.shape[1]]=resized
        self.net.setInput(canvas.transpose(2,0,1)[None])
        prediction=self.net.forward()[0]
        probabilities=prediction[:,4,None]*prediction[:,5:]
        classes=probabilities.argmax(1);confidence=probabilities.max(1)
        center=(prediction[:,:2]+self.anchors)*self.scales
        size=np.exp(np.clip(prediction[:,2:4],-20,20))*self.scales
        xywh=np.column_stack((center-size/2,size))/ratio
        # People, sports balls, rackets. Other moving objects come from residual motion.
        keep=np.flatnonzero((confidence>=self.confidence)&np.isin(classes,[0,32,38]))
        if not len(keep): return []
        selected=cv2.dnn.NMSBoxesBatched(xywh[keep].tolist(),confidence[keep].tolist(),classes[keep].tolist(),self.confidence,.45)
        result=[]
        for j in np.asarray(selected).reshape(-1):
            i=keep[j];x,y,bw,bh=xywh[i]
            box=[max(0,int(x)),max(0,int(y)),min(w,int(x+bw)),min(h,int(y+bh))]
            if box[2]-box[0]<3 or box[3]-box[1]<3: continue
            result.append({'bbox':box,'kind':{0:'person',32:'sports_ball',38:'tennis_racket'}[int(classes[i])],
                           'confidence':float(confidence[i]),'method':'yolox'})
        return result


def iou(a,b):
    area=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return area/max(1,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area)


class Appearance:
    """Frozen OSNet x0.25 MSMT17 person-ReID + clothing color, no face recognition."""
    def __init__(self, weights):
        import torch
        from .vendor.osnet import OSNet, OSBlock
        if sha(weights)!=OSNET_SHA:raise ValueError('Unexpected OSNet weight checksum')
        state=torch.load(weights,map_location='cpu',weights_only=True)
        self.encoder=OSNet(state['classifier.weight'].shape[0],blocks=[OSBlock]*3,layers=[2]*3,channels=[16,64,96,128])
        self.encoder.load_state_dict(state,strict=True);self.encoder.eval().requires_grad_(False)

    def __call__(self,image,boxes):
        import torch
        if not boxes:return []
        crops=[image[y1:y2,x1:x2] for x1,y1,x2,y2 in boxes]
        batch=torch.from_numpy(np.stack([cv2.resize(cv2.cvtColor(c,cv2.COLOR_BGR2RGB),(128,256)).transpose(2,0,1) for c in crops])).float()/255
        batch=(batch-torch.tensor([.485,.456,.406])[None,:,None,None])/torch.tensor([.229,.224,.225])[None,:,None,None]
        with torch.inference_mode(): features=self.encoder(batch).numpy()
        result=[]
        for crop,feature in zip(crops,features):
            # Exclude most head/background pixels from clothing histogram.
            h,w=crop.shape[:2];body=crop[int(h*.25):max(int(h*.85),int(h*.25)+1),int(w*.15):max(int(w*.85),int(w*.15)+1)]
            hist=cv2.calcHist([cv2.cvtColor(body,cv2.COLOR_BGR2HSV)],[0,1],None,[12,8],[0,180,0,256]).ravel()
            feature=feature/max(np.linalg.norm(feature),1e-8);hist=hist/max(np.linalg.norm(hist),1e-8)
            result.append(np.concatenate([np.sqrt(.85)*feature,np.sqrt(.15)*hist]))
        return result


class Tracker:
    def __init__(self,max_gap): self.tracks={};self.counter=0;self.max_gap=max_gap

    def update(self,detections,features,frame,segment):
        candidates=[]
        for i,(d,f) in enumerate(zip(detections,features)):
            for tid,t in self.tracks.items():
                if t['segment_id']!=segment or frame-t['last_frame']>self.max_gap or t['kind']!=d['kind']:continue
                overlap=iou(d['bbox'],t['bbox']);similarity=float(np.dot(f,t['feature']))
                a=np.array(d['bbox']);b=np.array(t['bbox'])
                distance=np.linalg.norm((a[:2]+a[2:]-b[:2]-b[2:])/2)/max(10,np.linalg.norm(b[2:]-b[:2]))
                if similarity>=.55 and (overlap>=.1 or distance<.8):candidates.append((.5*similarity+.5*overlap,i,tid))
        assigned={};used=set()
        for _,i,tid in sorted(candidates,reverse=True):
            if i not in assigned and tid not in used:assigned[i]=tid;used.add(tid)
        output=[]
        for i,(d,f) in enumerate(zip(detections,features)):
            tid=assigned.get(i)
            if tid is None:
                tid=f'track_{self.counter:05d}';self.counter+=1
                self.tracks[tid]={'id':tid,'segment_id':segment,'kind':d['kind'],'start_frame':frame,
                    'last_frame':frame,'bbox':d['bbox'],'feature':f,'samples':0,'best_confidence':0.,'moving_observations':0}
            t=self.tracks[tid];t['last_frame']=frame;t['bbox']=d['bbox'];t['samples']+=1
            t['feature']=t['feature']*.8+f*.2;t['feature']/=max(np.linalg.norm(t['feature']),1e-8)
            t['moving_observations']+=int(d['moving'])
            output.append({**d,'track_id':tid})
        return output


def associate(tracks, threshold=.78, margin=.08):
    """Mutual, unambiguous nearest neighbors across nonoverlapping shots. No name transfer."""
    links=[]
    groups={sid:[t for t in tracks if t['segment_id']==sid and t['samples']>=2] for sid in {t['segment_id'] for t in tracks}}
    ordered=sorted(groups)
    for ai,sid in enumerate(ordered):
        for other in ordered[ai+1:]:
            a,b=groups[sid],groups[other]
            if not a or not b:continue
            score=np.array([[float(np.dot(x['feature'],y['feature'])) if x['kind']==y['kind'] else -1 for y in b] for x in a])
            for i,x in enumerate(a):
                j=int(score[i].argmax());value=float(score[i,j]);alternatives=np.delete(score[i],j)
                reverse=np.delete(score[:,j],i)
                gap=value-max(float(alternatives.max()) if len(alternatives) else -1,float(reverse.max()) if len(reverse) else -1)
                if value>=threshold and gap>=margin and int(score[:,j].argmax())==i:
                    links.append({'track_a':x['id'],'track_b':b[j]['id'],'similarity':value,'margin':gap,
                                  'status':'appearance_candidate','name_transfer_allowed':False})
    return links


def association_sheet(directory):
    """Visual evidence for proposed links; similarity labels are not probabilities."""
    directory=Path(directory);tracks={t['id']:t for t in read_json(directory/'tracks.json')}
    links=read_json(directory/'associations.json')
    if not links:return None
    canvas=np.full((80+230*int(np.ceil(len(links)/3)),960,3),25,np.uint8)
    cv2.putText(canvas,'Cross-shot appearance candidates - names unconfirmed',(15,32),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1)
    cv2.putText(canvas,'Similarity is not an identity probability',(15,58),cv2.FONT_HERSHEY_SIMPLEX,.5,(180,180,180),1)
    for i,link in enumerate(links):
        left=(i%3)*320;top=80+(i//3)*230
        for side,key in enumerate(('track_a','track_b')):
            t=tracks[link[key]];crop=cv2.imread(str(directory/t['crop']))
            scale=min(135/crop.shape[1],165/crop.shape[0]);crop=cv2.resize(crop,(max(1,round(crop.shape[1]*scale)),max(1,round(crop.shape[0]*scale))))
            x=left+side*155+10;y=top+20;canvas[y:y+crop.shape[0],x:x+crop.shape[1]]=crop
            cv2.putText(canvas,t['id'],(x,top+12),cv2.FONT_HERSHEY_SIMPLEX,.4,(220,220,220),1)
            cv2.putText(canvas,f"f{t['crop_frame']}",(x,top+202),cv2.FONT_HERSHEY_SIMPLEX,.4,(220,220,220),1)
        cv2.putText(canvas,f"similarity {link['similarity']:.3f}",(left+10,top+222),cv2.FONT_HERSHEY_SIMPLEX,.45,(80,230,230),1)
    path=directory/'association-review.jpg';cv2.imwrite(str(path),canvas)
    return path


def residual_objects(previous,current,excluded):
    """Small foreground changes are moving-object candidates, never asserted to be a ball."""
    mask=(cv2.absdiff(previous,current)>28).astype(np.uint8)*255
    for x1,y1,x2,y2 in excluded:mask[max(0,y1-4):y2+4,max(0,x1-4):x2+4]=0
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
    count,_,stats,_=cv2.connectedComponentsWithStats(mask)
    result=[]
    for x,y,w,h,area in stats[1:]:
        if 5<=area<=1000 and w<=90 and h<=90:
            result.append({'bbox':[int(x),int(y),int(x+w),int(y+h)],'kind':'moving_object',
                           'confidence':None,'method':'frame_residual','moving':True,'track_id':None})
    return sorted(result,key=lambda d:(d['bbox'][2]-d['bbox'][0])*(d['bbox'][3]-d['bbox'][1]),reverse=True)[:25]


def score_signature(crop):
    hsv=cv2.cvtColor(cv2.resize(crop,(160,48)),cv2.COLOR_BGR2HSV)
    return ((hsv[:,:,1]<100)&(hsv[:,:,2]>180)).astype(np.uint8)*255


class ScoreGate:
    """Invalidate immediately on a change; read only after a stable crop and cooldown."""
    def __init__(self, cooldown, threshold=.025, settle_frames=10):
        self.cooldown=cooldown;self.threshold=threshold;self.accepted=None
        self.pending=None;self.count=0;self.invalid=False;self.last_read=-100000;self.settle_frames=settle_frames

    def update(self,signature,frame):
        if self.accepted is None:
            self.accepted=signature.copy();self.last_read=frame;return 'read'
        delta=float((signature!=self.accepted).mean())
        if delta<=self.threshold and not self.invalid:return 'unchanged'
        if not self.invalid:
            self.invalid=True;self.pending=signature.copy();self.count=1;return 'invalidate'
        if self.pending is not None and float((signature!=self.pending).mean())<=self.threshold:self.count+=1
        else:self.count=1
        self.pending=signature.copy()
        if self.count>=self.settle_frames and frame-self.last_read>=self.cooldown:
            self.accepted=signature.copy();self.last_read=frame;self.invalid=False;return 'read'
        return 'pending'


def run(video,output,start=0,end=None,config=None,detector_weights='data/pretrained/object_detection_yolox_2022nov.onnx',appearance_weights='data/pretrained/osnet_x0_25_msmt17.pth'):
    import torch
    config=config or TrackingConfig();output=Path(output)
    if output.exists() and any(output.iterdir()):raise ValueError('Use a new tracking output directory')
    output.mkdir(parents=True);(output/'crops').mkdir();(output/'scoreboard').mkdir()
    if config.detect_every<1:raise ValueError('detect_every must be positive')
    pixel_box(config.scoreboard,640,360)
    torch.set_num_threads(4);cv2.setNumThreads(4);started=time.monotonic();source_sha=sha(video)
    frames,segments=scan(video,start,end,config);fps=frames[0]['source_fps']
    detector=Detector(detector_weights,config.confidence);appearance=Appearance(appearance_weights)
    tracker=Tracker(round(fps*config.max_track_gap_seconds));previous=None;last_segment=None;active=[];last_detection=-10000
    writer=None;detections_count=0;score_candidates=[];score_gate=ScoreGate(round(fps*3),settle_frames=max(3,round(fps*.4)))
    try:
        with (output/'frames.jsonl').open('w',encoding='utf-8') as manifest:
            for meta,image in video_frames(video,start,end):
                row=frames[meta['source_frame']-start];h,w=image.shape[:2];gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                if meta['decoded_sha256']!=row['decoded_sha256']:raise ValueError('Decode differs between passes')
                if writer is None:
                    writer=cv2.VideoWriter(str(output/'annotated.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
                    if not writer.isOpened():raise ValueError('Could not create annotated video')
                sid=row['segment_id'];objects=[]
                if sid!=last_segment:active=[];last_detection=-10000
                if sid is not None:
                    if meta['source_frame']-last_detection>=config.detect_every:
                        dets=detector(image);detections_count+=1
                        for d in dets:
                            x1,y1,x2,y2=d['bbox']
                            fraction=float((cv2.absdiff(previous[y1:y2,x1:x2],gray[y1:y2,x1:x2])>25).mean()) if previous is not None else 0.
                            d['moving']=fraction>.015;d['motion_fraction']=fraction
                        features=appearance(image,[d['bbox'] for d in dets]);active=tracker.update(dets,features,meta['source_frame'],sid)
                        last_detection=meta['source_frame']
                        for d in active:
                            t=tracker.tracks[d['track_id']]
                            if d['confidence']>t['best_confidence']:
                                x1,y1,x2,y2=d['bbox'];p=output/'crops'/f"{d['track_id']}.jpg"
                                cv2.imwrite(str(p),image[y1:y2,x1:x2]);t.update(best_confidence=d['confidence'],crop=p.relative_to(output).as_posix(),crop_frame=meta['source_frame'])
                    # Boxes between detector samples are explicitly held, never claimed as exact detections.
                    objects=[{**d,'observation':'detected' if meta['source_frame']==last_detection else 'held',
                              'box_source_frame':last_detection,'box_age_frames':meta['source_frame']-last_detection} for d in active]
                    if previous is not None and last_segment==sid:
                        objects+=residual_objects(previous,gray,[d['bbox'] for d in active]+[pixel_box(config.scoreboard,w,h)])
                # Score sampling is independent of camera state, but never continuously sends video.
                x1,y1,x2,y2=pixel_box(config.scoreboard,w,h);crop=image[y1:y2,x1:x2]
                small=score_signature(crop)
                score_action=score_gate.update(small,meta['source_frame'])
                if score_action in ('read','invalidate'):
                    p=output/'scoreboard'/f"{meta['source_frame']:010d}.jpg";cv2.imwrite(str(p),crop)
                    score_candidates.append({'source_frame':meta['source_frame'],'timestamp':meta['timestamp'],
                        'path':p.relative_to(output).as_posix(),'sha256':sha(p),'action':score_action,
                        'last_verified_frame':meta['source_frame']})
                elif score_action=='unchanged' and score_candidates:
                    score_candidates[-1]['last_verified_frame']=meta['source_frame']
                canvas=image.copy()
                for d in objects:
                    color=(0,220,255) if d['kind']=='moving_object' else ((70,230,70) if d['moving'] else (150,150,150))
                    x1,y1,x2,y2=d['bbox'];cv2.rectangle(canvas,(x1,y1),(x2,y2),color,2)
                    label=d['track_id'] or d['kind'];cv2.putText(canvas,label,(x1,max(18,y1-4)),cv2.FONT_HERSHEY_SIMPLEX,.42,color,1)
                cv2.putText(canvas,f"frame {meta['source_frame']} | {row['camera']['status']} | {sid or 'no stable shot'}",(15,h-15),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
                writer.write(canvas)
                record={**row,'frame_id':frame_key(source_sha,meta['source_frame']),'width':w,'height':h,'objects':objects}
                manifest.write(json.dumps(record)+'\n');previous=gray;last_segment=sid
                if (meta['source_frame']-start)%250==0: print('tracking frame',meta['source_frame'],flush=True)
    finally:
        if writer is not None:writer.release()
    tracks=list(tracker.tracks.values());links=associate(tracks,config.association_threshold,config.association_margin)
    serial=[{k:v for k,v in t.items() if k!='feature'} for t in tracks]
    report={'video':str(Path(video).resolve()),'source_sha256':source_sha,'frame_index_base':0,
            'source_fps':fps,'start_frame':start,'end_frame_exclusive':frames[-1]['source_frame']+1,
            'config':asdict(config),'frames':len(frames),'stable_frames':sum(r['segment_id'] is not None for r in frames),
            'camera_states':dict(Counter(r['camera']['status'] for r in frames)), 'segments':segments,
            'detector_calls':detections_count,'tracks':len(tracks),'appearance_links':len(links),
            'elapsed_seconds':time.monotonic()-started,'vlm_calls':0,
            'models':{'detector':'YOLOX-S COCO OpenCV ONNX','detector_sha256':sha(detector_weights),
                      'appearance':'OSNet x0.25 MSMT17 person-ReID + HSV','appearance_sha256':sha(appearance_weights)},
            'limitations':['Camera stability is threshold-based, not guaranteed zero motion.',
                'Held boxes can lag between detector samples; source frame and age are recorded.',
                'Appearance links are candidates, not confirmed identities; names are not inferred from faces.',
                'Small balls may be missed; residual motion candidates may include shadows or noise.']}
    write_json(output/'report.json',report);write_json(output/'tracks.json',serial);write_json(output/'associations.json',links)
    write_json(output/'score-candidates.json',score_candidates)
    association_sheet(output)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--video',required=True);p.add_argument('--output',required=True)
    p.add_argument('--start-frame',type=int,default=0);p.add_argument('--end-frame',type=int)
    p.add_argument('--detector-weights',default='data/pretrained/object_detection_yolox_2022nov.onnx')
    p.add_argument('--appearance-weights',default='data/pretrained/osnet_x0_25_msmt17.pth')
    p.add_argument('--detect-every',type=int,default=5);p.add_argument('--scoreboard',type=float,nargs=4,default=TrackingConfig.scoreboard)
    a=p.parse_args();print(json.dumps(run(a.video,a.output,a.start_frame,a.end_frame,
        TrackingConfig(detect_every=a.detect_every,scoreboard=tuple(a.scoreboard)),a.detector_weights,a.appearance_weights),indent=2))


if __name__=='__main__':main()
