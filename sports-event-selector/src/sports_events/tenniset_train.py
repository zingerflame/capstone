"""Reusable TenniSet pilot training, without VLM calls or match split leakage.

Use sports_events.tenniset_filtered for current delta-filtered preparation.
The prepare command here preserves the historical fixed-stride baseline only.
"""
import argparse
import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch

from .io import read_json, write_json, fingerprint
from .tenniset import file_hash
from .learning import discover
from .pretrained import load_encoder, extract_features
from .experiment3 import train_variant, evaluate, candidate_key

SPLITS = {'V006': 'train', 'V007': 'train', 'V008': 'train', 'V009': 'val', 'V010': 'test'}


def project_anchors(original_frames, retained_frames):
    """Map to unique nearest sampled indices; ties choose the earlier frame."""
    values = np.asarray(retained_frames)
    return sorted({int(np.abs(values - frame).argmin()) for frame in original_frames})


def prepare(sequences_path, output, clips_per_video=12, seconds=8, stride=2):
    if clips_per_video < 1 or seconds <= 0 or stride < 1: raise ValueError('Invalid sampling configuration')
    output = Path(output)
    if output.exists() and any(output.iterdir()): raise ValueError('Use a fresh prepared dataset directory')
    sequences_path = Path(sequences_path).resolve()
    records = read_json(sequences_path)
    if {r['video_id'] for r in records} != set(SPLITS): raise ValueError('All five source videos required')
    # Read verified completed sequences, not a pending conversion plan.
    report = read_json(sequences_path.parent / 'conversion-report.json')
    if not report['complete']: raise ValueError('Exact-frame extraction must complete first')
    for row in records:
        for image, digest in zip(row['image_paths'], row['image_sha256']):
            if file_hash(image) != digest: raise ValueError(f'Ground-truth image changed: {image}')
    summaries = []
    for video_id, split in SPLITS.items():
        rows = [r for r in records if r['video_id'] == video_id]
        exact_images = {frame: image for row in rows for frame, image in
                        zip(row['original_frame_numbers'], row['image_paths'])}
        source = rows[0]['source_video']
        if file_hash(source) != rows[0]['source_video_sha256']: raise ValueError('Source video changed')
        fps = rows[0]['source_fps']
        length = int(round(seconds * fps))
        use = [r for r in rows if r['event']['type'] == 'use']
        if len(use) != 1: raise ValueError('Pilot expects one USE interval per video')
        lo, hi = use[0]['original_interval']
        starts = np.linspace(lo, hi - length + 1, clips_per_video).astype(int).tolist()
        if len(set(starts)) != len(starts) or any(b-a < length for a,b in zip(starts, starts[1:])):
            raise ValueError('Pilot clips must not overlap')
        capture = cv2.VideoCapture(source)
        for clip_number, start in enumerate(starts):
            identifier = f'tenniset_{video_id}_{clip_number:03d}'
            directory = output / identifier
            (directory / 'frames').mkdir(parents=True)
            (directory / 'scoreboards').mkdir()
            # OpenCV seek is for input context only; exact GT PNGs were decoded sequentially.
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, start): raise ValueError('Frame seek failed')
            if abs(capture.get(cv2.CAP_PROP_POS_FRAMES)-start) > .5: raise ValueError('Inexact frame seek')
            retained, frames = [], []
            for relative in range(length):
                if not capture.grab(): raise ValueError('Truncated source clip')
                if relative % stride: continue
                ok, image = capture.retrieve()
                if not ok: raise ValueError('Cannot decode context frame')
                if start+relative in exact_images:
                    exact = cv2.imread(exact_images[start+relative])
                    if exact is None or not np.array_equal(image,exact):
                        raise ValueError(f'Context seek disagrees with exact frame: {video_id}:{start+relative}')
                i = len(frames); image_path=f'frames/{i:08d}.jpg'; board_path=f'scoreboards/{i:08d}.jpg'
                if not cv2.imwrite(str(directory/image_path), image): raise IOError('Frame write failed')
                if not cv2.imwrite(str(directory/board_path), np.zeros((32,96,3),np.uint8)): raise IOError('Board write failed')
                retained.append(start+relative)
                frames.append({'index':i,'source_frame':start+relative,'timestamp':relative/fps,
                    'broadcast_timestamp':(start+relative)/fps,'image':image_path,'scoreboard':board_path})
            events = []; excluded=[]
            for row in rows:
                if not row['training_eligible']: continue
                anchors=row['original_frame_numbers']
                if anchors[-1] < start or anchors[0] >= start+length: continue
                if anchors[0] < start or anchors[-1] >= start+length:
                    excluded.append(row['id']); continue
                indices=project_anchors(anchors,retained)
                if len(indices) not in (2,3): raise ValueError('Sampling collapsed an event below two unique frames')
                events.append({'frame_indices':indices,'label':row['event']['label'],
                    'sequence_id':row['id'],'original_frame_numbers':anchors,
                    'mapped_original_frames':[retained[i] for i in indices]})
            clip={'schema_version':1,'id':identifier,'sport':'tennis','split':split,
                'group_id':f'tenniset_{video_id}','scoreboard_roi':None,'source_fps':fps,
                'source_frame_count':length,'source_video':source,'frames':frames,
                'sampling':{'method':'fixed_original_frame_stride','stride':stride},
                'source_start_frame':start,'source_sequences_sha256':file_hash(sequences_path)}
            write_json(directory/'clip.json',clip)
            write_json(directory/'annotations.json',{'reviewed':True,'review_basis':'publisher_human_intervals',
                'derived_targets':True,'source_sequences':str(sequences_path),'events':events,
                'excluded_partial_anchor_sequences':excluded,
                'note':'reviewed denotes imported publisher annotations, not a new independent human audit.'})
            summaries.append({'id':identifier,'split':split,'frames':len(frames),'events':len(events),
                              'partial_sequences_excluded':excluded,'start':start,'seconds':seconds})
        capture.release()
        print(f'{video_id}: prepared {clips_per_video} independent-of-label intervals',flush=True)
    write_json(output/'preparation.json',{'splits':SPLITS,'clips':summaries,
        'sequences_sha256':file_hash(sequences_path),'sampling_stride':stride,
        'limitations':'Bounded pilot, not full-dataset training. Fixed 12.5fps at stride 2 replaces delta sampling for an unambiguous mapping.'})


def train(dataset, output, weights, cache, epochs=20, seed=17):
    if epochs < 1: raise ValueError('epochs must be positive')
    output=Path(output)
    if output.exists() and any(output.iterdir()): raise ValueError('Use a fresh training output')
    records=discover(dataset)
    if {c['split'] for _,c,_ in records} != {'train','val','test'}: raise ValueError('Three splits required')
    write_json(output/'protocol.json',{'dataset':str(Path(dataset).resolve()),'epochs':epochs,'seed':seed,
        'model':'frozen_resnet18_transformer','motion':True,'loss':'v2',
        'split_groups':SPLITS,'selection':'validation only; freeze before test feature extraction and predictions',
        'dataset_preparation':read_json(Path(dataset)/'preparation.json'),
        'dataset_projection_report':read_json(Path(dataset).parent/'sequence-frames/conversion-report.json')
            if (Path(dataset).parent/'sequence-frames/conversion-report.json').exists() else None,
        'records':[{'clip':c['id'],'split':c['split'],'clip_hash':fingerprint(c),'truth':t} for _,c,t in records]})
    encoder=load_encoder(weights)
    trainval=[]
    for directory,clip,truth in records:
        if clip['split']=='test': continue
        trainval.append((directory,clip,truth,extract_features(directory,clip,encoder,cache)))
        print(f'Features: {clip["id"]}',flush=True)
    winners=train_variant(trainval,output/'model',True,seed,epochs)
    choice=max(winners,key=lambda name:candidate_key(winners[name]))
    write_json(output/'frozen_selection.json',{'budget_arm':choice,**winners[choice]})
    test=[]
    for directory,clip,truth in records:
        if clip['split']=='test':
            test.append((directory,clip,truth,extract_features(directory,clip,encoder,cache)))
    result=evaluate(test,output/'model'/f'{choice}.pt',output/'test')
    write_json(output/'results.json',result)
    print('HELD-OUT:',result,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare',help='Legacy fixed-stride reproduction; use sports_events.tenniset_filtered for delta-filtered training');q.add_argument('--sequences',default='data/sequence-frames/sequences.json')
    q.add_argument('--output',default='data/tenniset_pilot/clips');q.add_argument('--clips-per-video',type=int,default=12)
    q.add_argument('--seconds',type=float,default=8);q.add_argument('--stride',type=int,default=2)
    q=sub.add_parser('train');q.add_argument('--dataset',default='data/tenniset_delta/clips')
    q.add_argument('--target-mode',choices=['center20','legacy_frames'],default='center20')
    q.add_argument('--output',default='runs/tenniset_center20_recall');q.add_argument('--epochs',type=int,default=20)
    q.add_argument('--profile',choices=['window_recall','center3'],default='window_recall')
    q.add_argument('--seed',type=int,default=17);q.add_argument('--threads',type=int,default=4)
    q.add_argument('--weights',default='data/pretrained/resnet18-f37072fd.pth');q.add_argument('--cache',default='data/tenniset_delta_features')
    a=p.parse_args()
    if a.command=='prepare':prepare(a.sequences,a.output,a.clips_per_video,a.seconds,a.stride)
    else:
        torch.set_num_threads(a.threads)
        if a.target_mode=='center20':
            from .center20 import train as train_centers
            train_centers(a.dataset,a.output,a.weights,a.cache,a.epochs,a.seed,profile=a.profile)
        else:train(a.dataset,a.output,a.weights,a.cache,a.epochs,a.seed)


if __name__=='__main__':main()
