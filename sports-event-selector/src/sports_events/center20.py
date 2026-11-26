"""Event-center supervision and consecutive 20-frame output for the TenniSet model."""
import argparse
import math
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .core import boundaries, vote
from .io import read_json, write_json, fingerprint
from .learning import discover
from .pretrained import FeatureSelector, feature_logits, load_encoder, extract_features, WEIGHTS_SHA256

LENGTH=20


def window_credit(predicted_center,truth_center):
    distance=abs(predicted_center-truth_center)
    return max(.3,1-.7*distance/9.5) if distance<=9.5 else 0.


def window_targets(n,truth):
    centers=np.arange(max(0,n-LENGTH+1),dtype=float)+9.5
    if not len(truth):return np.zeros(len(centers),dtype=np.float32)
    distance=np.abs(centers[:,None]-np.asarray(truth)[None,:]).min(axis=1)
    return np.where(distance<=9.5,1-.7*distance/9.5,0).astype(np.float32)


def window_metrics(predicted,truth):
    """One event per window: maximize covered truths, then total decayed credit."""
    p,t=sorted(predicted),sorted(truth)
    dp=[[(0,0.) for _ in range(len(t)+1)] for _ in range(len(p)+1)]
    for i,a in enumerate(p,1):
        for j,b in enumerate(t,1):
            choices=[dp[i-1][j],dp[i][j-1]];credit=window_credit(a,b)
            if credit>0:
                m,c=dp[i-1][j-1];choices.append((m+1,c+credit))
            dp[i][j]=max(choices)
    m,c=dp[-1][-1]
    return {'matched':m,'predicted':len(p),'truth':len(t),'credit_sum':c}


def aggregate_windows(rows):
    m=sum(r['matched'] for r in rows);p=sum(r['predicted'] for r in rows);t=sum(r['truth'] for r in rows)
    c=sum(r['credit_sum'] for r in rows)
    return {'matched':m,'predicted':p,'truth':t,'precision':m/p if p else 0.,
            'recall':m/t if t else 0.,'f1':2*m/(p+t) if p+t else 0.,
            'weighted_precision':c/p if p else 0.,'weighted_recall':c/t if t else 0.,
            'mean_matched_credit':c/m if m else None,'credit_sum':c}


def center_targets(n,truth):
    centers=np.arange(max(0,n-LENGTH+1),dtype=float)+9.5
    if not len(truth):return np.zeros(len(centers),dtype=np.float32)
    distance=np.abs(centers[:,None]-np.asarray(truth)[None,:]).min(axis=1)
    return np.where(distance<=3,1-distance/4,0).astype(np.float32)


def center_loss(logits,ids,valid,n,truth,profile='center3'):
    if n<LENGTH:raise ValueError('Training context must contain at least 20 frames')
    total=torch.zeros(n,device=logits.device).scatter_add(0,ids[valid],logits[valid])
    support=torch.zeros_like(total).scatter_add(0,ids[valid],torch.ones_like(logits[valid]))
    frame_logits=total/support.clamp_min(1)
    scores=(frame_logits[9:n-10]+frame_logits[10:n-9])/2
    if profile not in ('center3','window_recall'):raise ValueError('Unknown target profile')
    target=torch.as_tensor((window_targets if profile=='window_recall' else center_targets)(n,truth),device=logits.device)
    element=F.binary_cross_entropy_with_logits(scores,target,reduction='none')
    positive=target>0
    positive_weight=.75 if profile=='window_recall' else .5
    relevance=(positive_weight*element[positive].mean()+(1-positive_weight)*element[~positive].mean()) if positive.any() and (~positive).any() else element.mean()
    prob=scores.sigmoid()
    # Differentiable union of candidate windows approximates unique retained-frame cost.
    starts=torch.arange(len(scores),device=logits.device)
    frames=torch.arange(n,device=logits.device)
    covers=((frames[:,None]>=starts)&(frames[:,None]<starts+LENGTH)).float()
    expected_coverage=(1-torch.exp(covers@torch.log1p(-prob.clamp(max=1-1e-6)))).sum()
    desired=set(boundaries(n))
    for c in truth:
        start=max(0,min(n-LENGTH,round(c-9.5)))
        desired.update(range(start,start+LENGTH))
    excess_frames=F.relu(expected_coverage-len(desired))/n
    excess_mass=F.relu(prob.sum()-target.sum())/len(scores)
    penalty=.05 if profile=='window_recall' else .2
    loss=relevance+penalty*excess_frames+penalty*excess_mass
    return loss


def decode_centers(probabilities,n,threshold=.5,consensus=.5,event_budget=None,min_separation=3):
    _,agreement,confidence,support=vote(probabilities,n,threshold)
    candidates=[]
    for start in range(max(0,n-LENGTH+1)):
        a,b=start+9,start+10
        if min(support[a],support[b])<3:continue
        agree=float((agreement[a]+agreement[b])/2)
        score=float((confidence[a]+confidence[b])/2)
        if agree>=consensus and score>=threshold:
            candidates.append((agree,score,start))
    candidates.sort(key=lambda r:(-r[0],-r[1],r[2]))
    budget=len(candidates) if event_budget is None else max(0,int(event_budget))
    selected=[]
    for agree,score,start in candidates:
        if len(selected)>=budget:break
        if any(abs(start-r[2])<min_separation for r in selected):continue
        selected.append((agree,score,start))
    selected.sort(key=lambda r:r[2])
    sequences=[list(range(start,start+LENGTH)) for _,_,start in selected]
    forced=boundaries(n)
    return {'output_mode':'center20','event_sequences':sequences,
            'event_centers':[s[0]+9.5 for s in sequences],
            'selected_indices':sorted(set(forced+[i for s in sequences for i in s])),
            'forced_indices':forced,'event_budget':budget,
            'center_scores':[{'agreement':a,'confidence':p} for a,p,_ in selected],
            'short_context':n<LENGTH}


def center_metrics(predicted,truth,tolerance=3):
    """One-to-one maximum matches, then minimum total center distance; retain duplicates."""
    p,t=sorted(predicted),sorted(truth)
    dp=[[(0,0.,[]) for _ in range(len(t)+1)] for _ in range(len(p)+1)]
    key=lambda x:(x[0],-x[1])
    for i,a in enumerate(p,1):
        for j,b in enumerate(t,1):
            options=[dp[i-1][j],dp[i][j-1]]
            if abs(a-b)<=tolerance:
                m,d,pairs=dp[i-1][j-1]
                options.append((m+1,d+abs(a-b),pairs+[(a,b)]))
            dp[i][j]=max(options,key=key)
    m,d,pairs=dp[-1][-1]
    precision=m/len(p) if p else 0.;recall=m/len(t) if t else 0.
    return {'matched':m,'predicted':len(p),'truth':len(t),'precision':precision,'recall':recall,
            'f1':2*m/(len(p)+len(t)) if p or t else 0.,
            'matched_center_mae':d/m if m else None,'center_distance_sum':d,
            'matches':[{'predicted':a,'truth':b,'distance':abs(a-b)} for a,b in pairs]}


def aggregate(metrics):
    m=sum(r['matched'] for r in metrics);p=sum(r['predicted'] for r in metrics);t=sum(r['truth'] for r in metrics)
    d=sum(r['center_distance_sum'] for r in metrics)
    return {'matched':m,'predicted':p,'truth':t,'precision':m/p if p else 0.,'recall':m/t if t else 0.,
            'f1':2*m/(p+t) if p+t else 0.,'matched_center_mae':d/m if m else None}


def truth_centers(directory,clip,source):
    frames=clip['frames']
    # Clip indices must represent consecutive members of the global filtered stream.
    if any(b['filtered_index_global']!=a['filtered_index_global']+1 for a,b in zip(frames,frames[1:])):
        raise ValueError('Non-consecutive filtered context')
    originals=np.array([f['source_frame'] for f in frames],float)
    truth=[]
    for e in read_json(Path(directory)/'annotations.json')['events']:
        r=source[e['sequence_id']]
        original_center=sum(r['original_interval'])/2
        if not originals[0]<=original_center<=originals[-1]:raise ValueError('Truth center outside context')
        # No rounding to predicted centers and no use of the augmented target triplet center.
        truth.append(float(np.interp(original_center,originals,np.arange(len(frames)))))
    return truth


def predict(model,state,features,clip):
    frames=clip['frames']
    if any('filtered_index_global' not in f for f in frames) or any(
        b['filtered_index_global']!=a['filtered_index_global']+1 for a,b in zip(frames,frames[1:])):
        raise ValueError('center20 inference requires consecutive delta-filtered frames')
    with torch.no_grad():logits,_,_=feature_logits(model,features)
    budget=math.ceil(state['event_rate']*len(features)*state['budget_factor'])
    result=decode_centers(logits.sigmoid().numpy(),len(features),state['threshold'],state['consensus'],budget)
    result['target_profile']=state.get('target_profile','center3')
    result['clip_id']=clip['id']
    result['event_sequence_frames']=[[clip['frames'][i] for i in seq] for seq in result['event_sequences']]
    result['center_timestamps_seconds']=[float(np.interp(c,np.arange(len(features)),
        [f['broadcast_timestamp'] for f in clip['frames']])) for c in result['event_centers']]
    return result


def train(dataset,output,weights,cache,epochs=20,seed=17,source='data/tenniset_delta/sequence-frames/sequences.json',profile='window_recall'):
    if epochs<1:raise ValueError('epochs must be positive')
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise ValueError('Use a new output directory')
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    records=discover(dataset);rows={r['id']:r for r in read_json(source)}
    if {c['split'] for _,c,_ in records}!={'train','val','test'}:raise ValueError('Require three match-separated splits')
    encoder=load_encoder(weights);prepared=[];snapshot=[]
    for directory,clip,_ in records:
        truth=truth_centers(directory,clip,rows)
        snapshot.append({'clip':clip['id'],'split':clip['split'],'truth_centers':truth,
                         'clip_hash':fingerprint(clip),'edge_centers_outside_output_range':sum(c<9.5 or c>len(clip['frames'])-10.5 for c in truth)})
        if clip['split']!='test':prepared.append((clip,truth,extract_features(directory,clip,encoder,cache)))
    training=[r for r in prepared if r[0]['split']=='train'];validation=[r for r in prepared if r[0]['split']=='val']
    rate=sum(len(t) for _,t,_ in training)/sum(len(f) for _,_,f in training)
    write_json(output/'protocol.json',{'output_mode':'center20','seed':seed,'epochs':epochs,
        'truth':'original human interval midpoint projected continuously into filtered index coordinates',
        'model':'frozen ResNet18 + motion + 10-frame Transformer', 'event_rate':rate,
        'target_profile':profile,'selection':'validation recall, weighted recall, precision, fewer predictions' if profile=='window_recall' else 'validation center F1 at ±3',
        'credit_rule':'1 at center, linear decay to 0.3 at distance 9.5, zero outside window' if profile=='window_recall' else 'legacy ±3',
        'positive_region_weight':.75 if profile=='window_recall' else .5,
        'excess_frame_and_mass_weights':.05 if profile=='window_recall' else .2,
        'records':snapshot,'source':str(Path(source).resolve()),'source_fingerprint':fingerprint(list(rows.values())),
        'limitations':'Reused pilot test, derived interval midpoints, 4 validation events, limited 8-second context.'})
    model=FeatureSelector(motion=True);optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-3)
    best=None;history=[]
    for epoch in range(1,epochs+1):
        model.train();order=list(training);random.shuffle(order);losses=[]
        for clip,truth,features in order:
            optimizer.zero_grad(set_to_none=True)
            logits,ids,valid=feature_logits(model,features)
            loss=center_loss(logits,ids,valid,len(features),truth,profile)
            if not torch.isfinite(loss):raise ValueError('Non-finite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();losses.append(float(loss.detach()))
        model.eval();cached=[]
        with torch.no_grad():
            for clip,truth,features in validation:
                logits,_,_=feature_logits(model,features)
                cached.append((logits.sigmoid().numpy(),len(features),truth))
        candidates=[]
        recall_mode=profile=='window_recall'
        for factor in ((1.,2.,3.,6.) if recall_mode else (1.,2.,3.)):
            for threshold in ((.05,.1,.2,.3,.5) if recall_mode else (.1,.3,.5,.7)):
                for consensus in ((.25,.5,.8) if recall_mode else (.5,.8)):
                    metric=window_metrics if recall_mode else center_metrics
                    metrics=[metric(decode_centers(p,n,threshold,consensus,math.ceil(rate*n*factor))['event_centers'],t) for p,n,t in cached]
                    candidates.append({'budget_factor':factor,'threshold':threshold,'consensus':consensus,**(aggregate_windows(metrics) if recall_mode else aggregate(metrics))})
        key=(lambda r:(r['recall'],r['weighted_recall'],r['precision'],-r['predicted'],-r['budget_factor'],r['threshold'],r['consensus'])) if recall_mode else (lambda r:(r['f1'],-r['predicted'],-r['budget_factor'],r['threshold'],r['consensus']))
        winner=max(candidates,key=key)
        if best is None or key(winner)>key(best):
            best={**winner,'epoch':epoch}
            torch.save({'model':model.state_dict(),'output_mode':'center20','motion':True,'width':model.width,
                        'event_rate':rate,'target_profile':profile,'weights_sha256':WEIGHTS_SHA256,**best},output/'model.pt')
        history.append({'epoch':epoch,'loss':float(np.mean(losses)),'validation':winner})
        write_json(output/'history.json',history)
        print(f"epoch={epoch} loss={np.mean(losses):.4f} validation recall={winner['recall']:.3f} F1={winner['f1']:.3f}",flush=True)
    write_json(output/'frozen_selection.json',best)
    state=torch.load(output/'model.pt',weights_only=True,map_location='cpu');model.load_state_dict(state['model']);model.eval()
    results=[]
    for directory,clip,_ in records:
        if clip['split']!='test':continue
        truth=truth_centers(directory,clip,rows);features=extract_features(directory,clip,encoder,cache)
        result=predict(model,state,features,clip);write_json(output/'predictions'/(clip['id']+'.json'),result)
        results.append({'clip':clip['id'],'frames':len(features),'selected':len(result['selected_indices']),
                        'window_metrics':window_metrics(result['event_centers'],truth),
                        'metrics':{str(d):center_metrics(result['event_centers'],truth,d) for d in (0,.5,1,2,3)}})
    summary={str(d):aggregate([r['metrics'][str(d)] for r in results]) for d in (0,.5,1,2,3)}
    summary['retained_fraction']=sum(r['selected'] for r in results)/sum(r['frames'] for r in results)
    summary['window_metrics']=aggregate_windows([r['window_metrics'] for r in results])
    write_json(output/'results.json',summary);write_json(output/'test-details.json',results)
    print(json.dumps(summary,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',default='data/tenniset_delta/clips')
    p.add_argument('--output',default='runs/tenniset_center20_recall');p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--profile',choices=['window_recall','center3'],default='window_recall')
    p.add_argument('--weights',default='data/pretrained/resnet18-f37072fd.pth');p.add_argument('--cache',default='data/tenniset_delta_features')
    args=p.parse_args();torch.set_num_threads(4);train(args.dataset,args.output,args.weights,args.cache,args.epochs,profile=args.profile)


if __name__=='__main__':
    main()
