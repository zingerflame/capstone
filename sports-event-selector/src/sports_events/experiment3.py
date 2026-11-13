"""Pretrained appearance, feature differences and budget ablations across three seeds."""
import argparse
import random
from pathlib import Path

import numpy as np
import torch

from .core import boundaries, decode, match_metrics
from .io import read_json, write_json, fingerprint
from .learning import discover, selection_loss
from .pilot import aggregate, import_teacher, benchmark
from .pretrained import WEIGHTS_SHA256, FeatureSelector, load_encoder, extract_features, feature_logits

SEEDS=(17,29,43)
THRESHOLDS=(.1,.2,.3,.4,.5,.6,.7)
CONSENSUS=(.5,.65,.8)
FACTORS=(1.,2.,3.)


def budget_for(n,rate,factor):
    return min(n,len(boundaries(n))+round(rate*factor*(n-len(boundaries(n)))))


def candidate_key(row):
    return (row['f1'],-row['predicted'],-row['budget_factor'],row['threshold'],row['consensus'])


def calibrate(model,records,rate):
    model.eval(); cached=[]
    with torch.no_grad():
        for _,clip,truth,features in records:
            if clip['split']!='val':continue
            logits,_,_=feature_logits(model,features)
            cached.append((logits.sigmoid().numpy(),len(features),truth))
    if not cached:raise ValueError('Validation data required')
    candidates=[]
    for factor in FACTORS:
        for threshold in THRESHOLDS:
            for consensus in CONSENSUS:
                metrics=[]
                for p,n,truth in cached:
                    result=decode(p,n,budget_for(n,rate,factor),threshold,consensus)
                    metrics.append(match_metrics([i for s in result['event_sequences'] for i in s],truth))
                candidates.append({'budget_factor':factor,'threshold':threshold,'consensus':consensus,
                                   **aggregate(metrics)})
    return {name:max([r for r in candidates if name=='expanded' or r['budget_factor']==1],key=candidate_key)
            for name in ('fixed','expanded')},candidates


def train_variant(records,output,motion,seed,epochs):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    training=[r for r in records if r[1]['split']=='train']
    rate=sum(len(set(t)-set(boundaries(len(c['frames'])))) for _,c,t,_ in training)/sum(
        len(c['frames'])-len(boundaries(len(c['frames']))) for _,c,_,_ in training)
    model=FeatureSelector(motion=motion)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-3)
    best={};history=[]
    for epoch in range(1,epochs+1):
        model.train();order=list(training);random.shuffle(order);details=[]
        for _,clip,truth,features in order:
            optimizer.zero_grad(set_to_none=True)
            logits,ids,valid=feature_logits(model,features)
            loss,metrics=selection_loss(logits,ids,valid,len(features),truth,version='v2',
                                       count_weight=.2*min(1,max(0,(epoch-1)/5)))
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
            details.append({'clip':clip['id'],**metrics})
        winners,candidates=calibrate(model,records,rate)
        for name,winner in winners.items():
            if name not in best or candidate_key(winner)>candidate_key(best[name]):
                best[name]={**winner,'epoch':epoch}
                torch.save({'model':model.state_dict(),'architecture':'frozen_resnet18_transformer',
                    'motion':motion,'width':model.width,'seed':seed,'epoch':epoch,'loss_version':'v2',
                    'weights_sha256':WEIGHTS_SHA256,'budget_rate':rate,**winner},output/(name+'.pt'))
        history.append({'epoch':epoch,'training':details,'validation_winners':winners,
                        'validation_candidates':candidates})
        write_json(output/'history.json',history);write_json(output/'selection.json',best)
        if epoch%5==0 or epoch==1:
            print(f'motion={motion} seed={seed} epoch={epoch}: fixed val={winners["fixed"]["f1"]:.3f}, expanded={winners["expanded"]["f1"]:.3f}',flush=True)
    return best


def load_head(checkpoint):
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if state.get('weights_sha256')!=WEIGHTS_SHA256:raise ValueError('Feature backbone mismatch')
    model=FeatureSelector(state['motion'],state['width'])
    model.load_state_dict(state['model']);model.eval()
    return model,state


def predict_features(model,state,features,clip):
    with torch.no_grad():logits,_,_=feature_logits(model,features)
    result=decode(logits.sigmoid().numpy(),len(features),budget_for(len(features),state['budget_rate'],
                  state['budget_factor']),state['threshold'],state['consensus'])
    result['clip_id']=clip['id']
    result['event_sequence_frames']=[[clip['frames'][i] for i in seq] for seq in result['event_sequences']]
    result['selected_frames']=[clip['frames'][i] for i in result['selected_indices']]
    return result


def evaluate(records,checkpoint,output):
    model,state=load_head(checkpoint);rows=[]
    for _,clip,truth,features in records:
        if clip['split']!='test':continue
        result=predict_features(model,state,features,clip)
        write_json(Path(output)/'predictions'/(clip['id']+'.json'),result)
        prediction=[i for s in result['event_sequences'] for i in s]
        rows.append({'clip':clip['id'],'frames':len(features),'selected':len(result['selected_indices']),
                     'metrics':{str(d):match_metrics(prediction,truth,d) for d in range(4)}})
    if not rows:raise ValueError('No test clips')
    summary={str(d):aggregate([r['metrics'][str(d)] for r in rows]) for d in range(4)}
    summary['retained_fraction']=sum(r['selected'] for r in rows)/sum(r['frames'] for r in rows)
    write_json(Path(output)/'metrics.json',{'summary':summary,'clips':rows})
    return summary


def summarize_seeds(rows):
    return {metric:{'mean':float(np.mean([r['3'][metric] for r in rows])),
                    'std':float(np.std([r['3'][metric] for r in rows])),
                    'values':[r['3'][metric] for r in rows]}
            for metric in ('precision','recall','f1','predicted','matched')}


def run(args):
    output=Path(args.output)
    if output.exists() and any(output.iterdir()):raise ValueError('Use a fresh output directory')
    original=discover('data/tennis_pilot/clips')
    trainval=[r for r in original if r[1]['split']!='test']
    import_teacher('data/tennis_third/clips','examples/tennis_third_teacher.json')
    fresh=discover('data/tennis_third/clips')
    previous=discover('data/tennis_second/clips')+[r for r in original if r[1]['split']=='test']
    if {c['group_id'] for _,c,_ in trainval}&{c['group_id'] for _,c,_ in fresh+previous}:
        raise ValueError('Test match overlaps train/validation')
    protocol={'seeds':SEEDS,'epochs':args.epochs,'thresholds':THRESHOLDS,'consensus':CONSENSUS,
        'budget_factors':FACTORS,'training_seconds':72,'validation_seconds':24,'fresh_test_seconds':28,
        'architecture_arms':['appearance','motion'],'budget_arms':['fixed','expanded'],
        'primary_selection':'Highest mean validation F1 across all three seeds. Seed 17 is the fixed representative checkpoint, never selected on test.',
        'weights_sha256':WEIGHTS_SHA256,'teacher_fingerprint':fingerprint(read_json('examples/tennis_third_teacher.json')),
        'trainval_snapshot':[{'id':c['id'],'split':c['split'],'clip_hash':fingerprint(c),'truth':t} for _,c,t in trainval],
        'fresh_intervals':read_json('examples/tennis_third_selections.json'),
        'limitations':'Tiny pseudo-labeled data; new intervals in the same previously inspected held-out matches.'}
    write_json(output/'protocol.json',protocol)
    encoder=load_encoder(args.weights)
    train_features=[(d,c,t,extract_features(d,c,encoder,args.cache)) for d,c,t in trainval]
    selections={}
    for arm in ('appearance','motion'):
        for seed in SEEDS:
            selections[f'{arm}_{seed}']=train_variant(train_features,output/arm/str(seed),arm=='motion',seed,args.epochs)
    ranking=[]
    for arm in ('appearance','motion'):
        for budget in ('fixed','expanded'):
            ranking.append({'arm':arm,'budget':budget,'mean_validation_f1':float(np.mean([
                selections[f'{arm}_{seed}'][budget]['f1'] for seed in SEEDS]))})
    primary=max(ranking,key=lambda r:r['mean_validation_f1'])
    write_json(output/'frozen_selection.json',{'ranking':ranking,'primary':primary,'selections':selections})
    print('FROZEN before test:',primary,flush=True)
    result={'primary':primary,'validation_ranking':ranking,'fresh':{},'regression':{}}
    # Frozen encoder extraction is label-free. Test predictions begin only here.
    for name,records in (('fresh',fresh),('regression',previous)):
        prepared=[(d,c,t,extract_features(d,c,encoder,args.cache)) for d,c,t in records]
        result[name]['previous_model']=benchmark(records,'runs/tennis_second/v2/best.pt',output/name/'previous_model')
        for arm in ('appearance','motion'):
            for budget in ('fixed','expanded'):
                metrics=[]
                for seed in SEEDS:
                    metrics.append(evaluate(prepared,output/arm/str(seed)/(budget+'.pt'),output/name/arm/budget/str(seed)))
                result[name][arm+'_'+budget]={'seeds':metrics,'aggregate':summarize_seeds(metrics)}
    write_json(output/'comparison.json',result)
    print('Fresh results:',{k:v['aggregate']['f1'] for k,v in result['fresh'].items() if k!='previous_model'},flush=True)


def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('run','predict'):
        p=sub.add_parser(name)
        p.add_argument('--weights',default='data/pretrained/resnet18-f37072fd.pth')
        p.add_argument('--cache',default='data/tennis_features')
        p.add_argument('--threads',type=int,default=4)
        if name=='run':
            p.add_argument('--output',default='runs/tennis_third')
            p.add_argument('--epochs',type=int,default=20)
        else:
            p.add_argument('--checkpoint',required=True);p.add_argument('--clip',required=True);p.add_argument('--output',required=True)
    args=parser.parse_args();torch.set_num_threads(args.threads)
    if args.command=='run':
        if args.epochs<1:raise ValueError('epochs must be positive')
        run(args)
    else:
        directory=Path(args.clip);clip=read_json(directory/'clip.json')
        features=extract_features(directory,clip,load_encoder(args.weights),args.cache)
        model,state=load_head(args.checkpoint)
        write_json(args.output,predict_features(model,state,features,clip))


if __name__=='__main__':main()
