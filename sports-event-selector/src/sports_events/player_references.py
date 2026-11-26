"""Scoreboard-text detection and auditable player reference JPEG extraction.

Names come from readable text or explicit supplied metadata, never face recognition.
Detection proposes associations for review. A reviewed spec produces reference bytes.
"""
import argparse
from pathlib import Path
import re

import cv2
from .commentary import Provider
from .io import read_json, write_json
from .vertex_dataset import sha

DETECTION_PROMPT = '''Read visible player names from scoreboard text in this tennis frame.
Do not identify anyone from their face or appearance. Detect the two active court players,
excluding officials, spectators and ball staff. Return JSON:
{"scoreboard_names":["literal readable name"], "server_name_text":null,
"near_bbox":[x1,y1,x2,y2], "far_bbox":[x1,y1,x2,y2],
"serving_side":"near|far|unknown", "evidence":"explanation of scoreboard server marker and pose"}.
Use normalized 0..1 boxes. server_name_text must be readable scoreboard text supported
by a server marker, otherwise null. Do not guess truncated names or use appearance to name people.
Use null for an undetectable box. These outputs will be reviewed before creating named references.'''


def box_pixels(box, width, height):
    if (not isinstance(box,list) or len(box)!=4 or
        any(type(v) not in (int,float) for v in box) or
        not (0<=box[0]<box[2]<=1 and 0<=box[1]<box[3]<=1)):
        raise ValueError('Invalid normalized player box')
    x1,y1,x2,y2=[round(v*scale) for v,scale in zip(box,[width,height,width,height])]
    if x2-x1<8 or y2-y1<8:raise ValueError('Player crop too small')
    return x1,y1,x2,y2


def detect(image, provider, model, output):
    image=Path(image)
    im=cv2.imread(str(image))
    if im is None:raise ValueError('Unreadable source image')
    # Provider accepts JPEG parts. Reencode PNG input before calling.
    jpeg=Path(output).parent/'detector-input.jpg'
    jpeg.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(jpeg),im):raise ValueError('JPEG write failed')
    result=Provider(provider,model,Path(output).parent/'detector-calls').ask(DETECTION_PROMPT,[jpeg])
    names=result.get('scoreboard_names')
    if not isinstance(names,list) or any(not isinstance(n,str) or not n.strip() for n in names):
        raise ValueError('Invalid scoreboard OCR response')
    if result.get('server_name_text') is not None and result['server_name_text'] not in names:
        raise ValueError('Server name must be an observed scoreboard name')
    if result.get('serving_side') not in ('near','far','unknown'):raise ValueError('Invalid serving side')
    for key in ('near_bbox','far_bbox'):
        if result.get(key) is not None:box_pixels(result[key],im.shape[1],im.shape[0])
    proposals=[]
    if len(names)==2 and result.get('server_name_text') and result['serving_side']!='unknown':
        serving=result['serving_side'];other='far' if serving=='near' else 'near'
        for name,side in [(result['server_name_text'],serving),(next(n for n in names if n!=result['server_name_text']),other)]:
            if result.get(side+'_bbox') is not None:
                proposals.append({'name':name,'bbox':result[side+'_bbox'],'reviewed':False,
                                  'identity_basis':'scoreboard server marker plus observed serving side'})
    result.update(source_image=str(image.resolve()),source_sha256=sha(image),proposals=proposals,
                  status='needs_review',provider=provider,model=model)
    write_json(output,result)
    return result


def build_references(specification, output):
    spec=read_json(specification);output=Path(output);registry={}
    for video, entries in spec.items():
        if not re.fullmatch(r'V\d+',video):raise ValueError('Invalid video ID')
        registry[video]=[]
        names=set()
        for entry in entries:
            if not entry.get('reviewed') or not entry.get('identity_basis'):
                raise ValueError('Each reference needs reviewed identity provenance')
            if not isinstance(entry.get('name'),str) or not entry['name'].strip() or entry['name'] in names:
                raise ValueError('Invalid or duplicate player name')
            names.add(entry['name'])
            image=cv2.imread(entry['source_image'])
            if image is None:raise ValueError('Missing reference source')
            if sha(entry['source_image'])!=entry['source_sha256']:raise ValueError('Reference source changed')
            x1,y1,x2,y2=box_pixels(entry['bbox'],image.shape[1],image.shape[0])
            crop=image[y1:y2,x1:x2]
            # Preserve native resolution; no invented high-resolution face details.
            rel=f'assets/players/{video}/player_{len(registry[video])+1:02}.jpg'
            destination=output/rel;destination.parent.mkdir(parents=True,exist_ok=True)
            if not cv2.imwrite(str(destination),crop,[cv2.IMWRITE_JPEG_QUALITY,95]):raise ValueError('Crop write failed')
            registry[video].append({**entry,'path':rel,'sha256':sha(destination),
                                   'width':x2-x1,'height':y2-y1})
    write_json(output/'players.json',registry)
    print(f'Created {sum(map(len,registry.values()))} player references across {len(registry)} videos.')
    return registry


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['detect','build'])
    p.add_argument('--image');p.add_argument('--provider',choices=['gpt','gemini'],default='gemini')
    p.add_argument('--model');p.add_argument('--specification')
    p.add_argument('--output',default='data/tenniset_vertex20')
    args=p.parse_args()
    if args.command=='detect':
        if not args.image or not args.model:p.error('detect requires --image and --model')
        detect(args.image,args.provider,args.model,args.output)
    else:
        if not args.specification:p.error('build requires --specification')
        build_references(args.specification,args.output)


if __name__=='__main__':main()
