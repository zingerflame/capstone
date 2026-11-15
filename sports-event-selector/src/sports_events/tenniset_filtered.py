"""Filter each entire video first, then project/augment TenniSet targets.

Sampling is exactly video.filtered_frames(), independent of labels. Annotation
access controls which retained images are materialized, never which frames survive.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from .io import read_json, write_json, fingerprint
from .tenniset import file_hash
from .tenniset_train import SPLITS
from .video import filtered_frames


def map_targets(anchors, interval, retained, fps, max_gap_seconds=.16):
    """Closest unique frames, augmented to three when bounded candidates exist."""
    values = np.asarray(retained, dtype=np.int64)
    anchors = np.asarray(anchors, dtype=np.int64)
    if len(values)==0 or np.any(np.diff(values)<=0) or fps<=0 or max_gap_seconds<0:
        raise ValueError('Invalid retained feed or mapping limits')
    lo=max(interval[0],int(anchors.min()-max_gap_seconds*fps))
    hi=min(interval[1],int(anchors.max()+max_gap_seconds*fps))
    candidates=np.arange(np.searchsorted(values,lo),np.searchsorted(values,hi,side='right'))
    if len(candidates):
        distances=np.abs(values[candidates,None]-anchors[None,:]).min(axis=1)
        candidates=candidates[distances<=max_gap_seconds*fps+1e-9]
    if not len(candidates):
        return {'status':'unmappable','filtered_frame_numbers':[], 'reason':'no_retained_frame_within_bounds'}
    nearest=[int(min(candidates,key=lambda j:(abs(int(values[j])-int(a)),int(values[j])))) for a in anchors]
    chosen=set(nearest)
    initial=len(chosen)
    order=sorted(candidates,key=lambda j:(min(abs(int(values[j])-int(a)) for a in anchors),
                                        abs(float(values[j])-float(anchors.mean())),int(values[j])))
    for j in order:
        if len(chosen)>=min(3,len(candidates)):break
        chosen.add(int(j))
    chosen=sorted(chosen)
    if len(chosen)<2:
        return {'status':'unmappable','filtered_frame_numbers':[], 'reason':'fewer_than_two_distinct_retained_frames',
                'nearest_unique_count':initial}
    return {'status':'mapped','filtered_frame_numbers':chosen,
        'original_frame_numbers':[int(values[j]) for j in chosen],
        'nearest_unique_count':initial,'augmented_count':len(chosen)-initial,
        'distance_to_original_target_seconds':[min(abs(int(values[j])-int(a)) for a in anchors)/fps for j in chosen]}


def merged_intervals(intervals):
    result=[]
    for lo,hi in sorted(intervals):
        if result and lo<=result[-1][1]+1:result[-1][1]=max(hi,result[-1][1])
        else:result.append([lo,hi])
    return result


def scan_video(video_id, rows, specs, output, sampling, max_gap_seconds):
    output=Path(output).resolve(); source=rows[0]['source_video'];fps=rows[0]['source_fps']
    digest=file_hash(source)
    if digest!=rows[0]['source_video_sha256']:raise ValueError('Source video changed')
    signature=fingerprint({'source':digest,'sampling':sampling,'specs':specs,'rows':rows,
                           'max_gap_seconds':max_gap_seconds,'recipe':'full_feed_delta_then_projection_v1'})
    receipt=output/'feeds'/f'{video_id}.json'
    if receipt.exists():
        state=read_json(receipt)
        if state['signature']!=signature:raise ValueError('Changed filter inputs: use a new output directory')
        return state
    padding=max_gap_seconds*fps
    ranges=merged_intervals([(max(r['original_interval'][0],int(min(r['original_frame_numbers'])-padding)),
                              min(r['original_interval'][1],int(max(r['original_frame_numbers'])+padding))) for r in rows])
    directory=output/'sequence-frames'/video_id;directory.mkdir(parents=True,exist_ok=True)
    clip_records={s['id']:[] for s in specs}
    for spec in specs:
        clip_dir=output/'clips'/spec['id']
        (clip_dir/'frames').mkdir(parents=True,exist_ok=True)
        (clip_dir/'scoreboards').mkdir(exist_ok=True)
    capture=cv2.VideoCapture(source)
    if not capture.isOpened() or abs(capture.get(cv2.CAP_PROP_FPS)-fps)>1e-6:
        capture.release();raise ValueError('Source FPS does not match annotation provenance')
    frames=[]; images={}; pointer=0; source_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    try:
        for global_index,(image,original,timestamp) in enumerate(filtered_frames(capture,**sampling)):
            frames.append({'filtered_index':global_index,'source_frame':original,'timestamp':timestamp})
            while pointer<len(ranges) and ranges[pointer][1]<original:pointer+=1
            if pointer<len(ranges) and ranges[pointer][0]<=original<=ranges[pointer][1]:
                path=directory/f'{original:010d}.png'
                if not cv2.imwrite(str(path),image,[cv2.IMWRITE_PNG_COMPRESSION,1]):raise IOError(path)
                restored=cv2.imread(str(path))
                if restored is None or not np.array_equal(restored,image):raise ValueError('PNG round-trip mismatch')
                images[str(original)]={'path':str(path),'sha256':file_hash(path)}
            for spec in specs:
                if spec['start']<=original<spec['start']+round(spec['seconds']*fps):
                    local=len(clip_records[spec['id']]);clip_dir=output/'clips'/spec['id']
                    scene=f'frames/{local:08d}.jpg';board=f'scoreboards/{local:08d}.jpg'
                    if not cv2.imwrite(str(clip_dir/scene),image):raise IOError('Scene write failed')
                    if not cv2.imwrite(str(clip_dir/board),np.zeros((32,96,3),np.uint8)):raise IOError('Board write failed')
                    clip_records[spec['id']].append({'index':local,'filtered_index_global':global_index,
                        'source_frame':original,'timestamp':timestamp-spec['start']/fps,
                        'broadcast_timestamp':timestamp,'image':scene,'scoreboard':board})
            if global_index and global_index%15000==0:
                print(f'{video_id}: filtered through original {original}/{source_count}',flush=True)
    finally:capture.release()
    state={'signature':signature,'source_video':source,'source_sha256':digest,'fps':fps,
        'source_frame_count':source_count,'sampling':sampling,'frames':frames,'images':images,
        'clip_frames':clip_records}
    write_json(receipt,state)
    print(f'{video_id}: {source_count} original -> {len(frames)} retained',flush=True)
    return state


def build(sequences_path, baseline_preparation, output, min_fps=10., max_fps=20., delta=.035,
          max_gap_seconds=.16, workers=2):
    if not 0<min_fps<=max_fps or not 0<=delta<=1 or max_gap_seconds<0 or workers<1:
        raise ValueError('Invalid filtering configuration')
    output=Path(output).resolve()
    records=read_json(sequences_path)
    if not read_json(Path(sequences_path).parent/'conversion-report.json')['complete']:
        raise ValueError('Original sequence conversion is incomplete')
    specs=read_json(baseline_preparation)['clips']
    if {r['video_id'] for r in records}!=set(SPLITS):raise ValueError('All five videos required')
    sampling={'min_fps':min_fps,'max_fps':max_fps,'delta':delta}
    cv2.setNumThreads(2)
    jobs=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for video_id in SPLITS:
            rows=[r for r in records if r['video_id']==video_id]
            selected=[s for s in specs if s['id'].startswith(f'tenniset_{video_id}_')]
            if not selected or any(s['split']!=SPLITS[video_id] for s in selected):raise ValueError('Invalid clip split')
            jobs.append((video_id,pool.submit(scan_video,video_id,rows,selected,output,sampling,max_gap_seconds)))
        feeds={name:job.result() for name,job in jobs}
    converted=[]
    for row in records:
        feed=feeds[row['video_id']]
        mapping=map_targets(row['original_frame_numbers'],row['original_interval'],
                            [f['source_frame'] for f in feed['frames']],feed['fps'],max_gap_seconds)
        original=mapping.get('original_frame_numbers',[])
        converted.append({**row,'source_sequence_id':row['id'],
            'original_sequence_frame_numbers':row['original_frame_numbers'],
            'original_sequence_image_paths':row['image_paths'],
            'original_frame_numbers':original,
            'filtered_frame_numbers':mapping['filtered_frame_numbers'],
            'image_paths':[feed['images'][str(f)]['path'] for f in original],
            'image_sha256':[feed['images'][str(f)]['sha256'] for f in original],
            'mapping':mapping,'sampling':sampling,'mapping_max_gap_seconds':max_gap_seconds,
            'selection_policy':'closest_unique_retained_frames_with_bounded_augmentation_v1',
            'training_eligible':row['training_eligible'] and mapping['status']=='mapped',
            'original_training_eligible':row['training_eligible']})
    summaries=[]
    for spec in specs:
        video_id=spec['id'].split('_')[1];feed=feeds[video_id];frames=feed['clip_frames'][spec['id']]
        directory=output/'clips'/spec['id'];end=spec['start']+round(spec['seconds']*feed['fps'])
        lookup={r['filtered_index_global']:r['index'] for r in frames};events=[];unresolved=[]
        for row in converted:
            if row['video_id']!=video_id or not row['original_training_eligible']:continue
            anchors=row['original_sequence_frame_numbers']
            if anchors[-1]<spec['start'] or anchors[0]>=end:continue
            ids=row['filtered_frame_numbers']
            if row['mapping']['status']!='mapped' or not all(i in lookup for i in ids):
                unresolved.append(row['id']);continue
            events.append({'frame_indices':[lookup[i] for i in ids],'label':row['event']['label'],
                'sequence_id':row['id'],'original_frame_numbers':row['original_frame_numbers'],
                'filtered_frame_numbers':ids,'original_sequence_frame_numbers':anchors,
                'mapping':row['mapping']})
        # Never silently turn an unresolved positive into a negative training example.
        if unresolved:
            write_json(directory/'excluded.json',{'unresolved_events':unresolved,'spec':spec})
        else:
            write_json(directory/'clip.json',{'schema_version':1,'id':spec['id'],'sport':'tennis',
                'split':spec['split'],'group_id':f'tenniset_{video_id}','scoreboard_roi':None,
                'source_fps':feed['fps'],'source_frame_count':feed['source_frame_count'],
                'source_video':feed['source_video'],'source_start_frame':spec['start'],
                'sampling':{**sampling,'scope':'entire_original_video','implementation':'video.filtered_frames'},
                'frames':frames,'source_sequences_sha256':file_hash(sequences_path)})
            write_json(directory/'annotations.json',{'reviewed':True,'derived_targets':True,
                'review_basis':'publisher_human_intervals_projected_after_actual_filtering','events':events,
                'source_sequences':str(output/'sequence-frames/sequences.json')})
        summaries.append({**spec,'frames':len(frames),'events':len(events),'excluded':bool(unresolved),
                          'unresolved_events':unresolved})
    write_json(output/'sequence-frames/sequences.json',converted)
    report={'complete':True,'records':len(converted),'mapped':sum(r['mapping']['status']=='mapped' for r in converted),
        'mapped_actions':sum(r['training_eligible'] for r in converted),
        'augmented_actions':sum(r['original_training_eligible'] and r['mapping'].get('augmented_count',0)>0 for r in converted),
        'collapsed_to_one_before_augmentation':sum(r['original_training_eligible'] and r['mapping'].get('nearest_unique_count')==1 for r in converted),
        'unmappable_action_ids':[r['id'] for r in converted if r['original_training_eligible'] and not r['training_eligible']],
        'excluded_clips':[r['id'] for r in summaries if r['excluded']],
        'sampling':sampling,'max_gap_seconds':max_gap_seconds,
        'videos':{name:{'original':f['source_frame_count'],'retained':len(f['frames'])} for name,f in feeds.items()},
        'comparison_note':'Same match split and original clip intervals as the earlier pilot. Reused test is a regression evaluation.'}
    write_json(output/'sequence-frames/conversion-report.json',report)
    write_json(output/'clips/preparation.json',{'splits':SPLITS,'clips':summaries,'sampling':sampling})
    print(report,flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sequences',default='data/sequence-frames/sequences.json')
    p.add_argument('--baseline-preparation',default='data/tenniset_pilot/clips/preparation.json')
    p.add_argument('--output',default='data/tenniset_delta')
    p.add_argument('--min-fps',type=float,default=10);p.add_argument('--max-fps',type=float,default=20)
    p.add_argument('--delta',type=float,default=.035);p.add_argument('--max-gap-seconds',type=float,default=.16)
    p.add_argument('--workers',type=int,default=2)
    a=p.parse_args();build(a.sequences,a.baseline_preparation,a.output,a.min_fps,a.max_fps,a.delta,a.max_gap_seconds,a.workers)


if __name__=='__main__':main()
