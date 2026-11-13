"""Render standalone diagrams from measured local artifacts (requires Pillow)."""
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'docs/diagrams'
OUT.mkdir(parents=True, exist_ok=True)
INK, MUTED = '#15283e', '#516477'
BLUE, GREEN, RED, ORANGE = '#1769aa', '#14765b', '#b73a48', '#aa6012'


def font(size, bold=False):
    candidates = [Path('C:/Windows/Fonts')/('segoeuib.ttf' if bold else 'segoeui.ttf'),
                  Path('/usr/share/fonts/truetype/dejavu')/('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')]
    for p in candidates:
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default(size=size)


def canvas(title, subtitle, height=760):
    image = Image.new('RGB',(1400,height),'#f5f8fc')
    draw = ImageDraw.Draw(image)
    draw.text((40,30),title,font=font(36,True),fill=INK)
    draw.text((40,88),subtitle,font=font(21),fill=MUTED)
    return image, draw


def lines(draw, text, xy, width, size=22, color=INK):
    x,y=xy
    for paragraph in text.split('\n'):
        current=''
        for word in paragraph.split():
            candidate=(current+' '+word).strip()
            if draw.textlength(candidate,font=font(size))>width and current:
                draw.text((x,y),current,font=font(size),fill=color); y+=size+9; current=word
            else: current=candidate
        draw.text((x,y),current,font=font(size),fill=color); y+=size+9
    return y


def box(draw, rect, title, body, color=BLUE):
    x,y,w,h=rect
    draw.rounded_rectangle((x,y,x+w,y+h),radius=16,fill='white',outline='#d5dfe9',width=2)
    draw.rounded_rectangle((x,y,x+6,y+h),radius=3,fill=color)
    draw.text((x+20,y+18),title,font=font(25,True),fill=color)
    bottom=lines(draw,body,(x+20,y+62),w-40)
    if bottom>y+h+3:
        raise ValueError(f'Box overflow: {title}')


def arrow(draw,start,end):
    draw.line((start,end),fill=MUTED,width=3)
    x,y=end
    if start[1]==y: draw.polygon([(x,y),(x-10,y-6),(x-10,y+6)],fill=MUTED)
    else: draw.polygon([(x,y),(x-6,y-10),(x+6,y-10)],fill=MUTED)


def main():
    im,d=canvas('01  What failed in the first pilot?', 'Execution succeeded. Event selection failed the original small held-out test.')
    for x,title,body in [(40,'Downloaded','10 full broadcasts\n14.95 hours of video'),
                         (380,'Teacher reviewed','128 seconds total\nChatGPT pseudo-labels'),
                         (720,'Student trained','72 seconds / 6 matches\n20 epochs from scratch'),
                         (1060,'Original test','32 seconds / 2 matches\n15 teacher frame anchors')]:
        box(d,(x,160,300,190),title,body)
        if x<1060:arrow(d,(x+305,255),(x+330,255))
    box(d,(40,415,640,240),'Software worked','Downloads, frame sampling, gradients, saved weights and prediction files all completed.\nLower loss did not establish tennis understanding.',GREEN)
    box(d,(720,415,640,240),'Predictions failed','10 proposed event frames; 0 matched within +/-3 filtered positions.\nPrecision 0%; recall 0%; F1 0.',RED)
    d.text((40,700),'Unreviewed source footage was never treated as negative training data.',font=font(22),fill=MUTED)
    im.save(OUT/'first-pilot.png')

    im,d=canvas('02  Two training instructions disagreed', 'Controlled example: the model exactly reproduces the requested distance-soft target.',800)
    values=['.25','.50','.75','1.0','1.0','1.0','.75','.50','.25']
    for k,value in enumerate(values):
        x=115+k*130
        d.rounded_rectangle((x,155,x+115,255),radius=10,fill='#e5effa' if k not in (3,4,5) else '#c8eadc')
        d.text((x+20,166),str(k+11),font=font(21),fill=MUTED)
        d.text((x+20,202),value,font=font(28,True),fill=INK)
    d.text((115,280),'Frames 11-14: mass 2.5',font=font(22,True),fill=INK)
    d.text((635,280),'15: 1.0',font=font(22,True),fill=INK)
    d.text((765,280),'Frames 16-19: mass 2.5',font=font(22,True),fill=INK)
    box(d,(40,360,640,245),'OLD: cap every anchor at 1','The soft target asks for 2.5 units near an edge anchor. The duplicate term allows only 1.\nEven this exact target pays a penalty of 1.5.',RED)
    box(d,(720,360,640,245),'NEW: allow the target mass','Caps become 2.5, 1.0 and 2.5 for this example. The exact target pays no duplicate penalty.\nAdditional probability mass is still penalized.',GREEN)
    lines(d,'Confirmed in all six positive training clips. This fixes one objective conflict; it does not prove the sole cause of failure. Actual selected-frame count is still constrained.',(40,660),1300,23,MUTED)
    im.save(OUT/'loss-conflict.png')

    im,d=canvas('03  Second experiment: a controlled comparison', 'Training data, architecture and frame-selection rules remain unchanged.',920)
    box(d,(40,140,1320,120),'Same inputs','6 training matches / 72 seconds. Same seed 17, optimizer, learning rate and 20 epochs.')
    arrow(d,(360,265),(360,305));arrow(d,(1040,265),(1040,305))
    box(d,(40,310,640,175),'Control','Original loss. Calibrate every epoch, instead of only two saved checkpoints.',ORANGE)
    box(d,(720,310,640,175),'Revised','Balanced event/background regions, corrected duplicate mass, gentler count penalty with warmup.',GREEN)
    arrow(d,(360,490),(360,535));arrow(d,(1040,490),(1040,535))
    box(d,(40,540,1320,125),'Validation only chooses checkpoints','Same 2 validation matches. Search 7 thresholds x 3 consensus values at every epoch.')
    arrow(d,(700,670),(700,710))
    box(d,(40,715,1320,135),'Freeze models, then evaluate','28 seconds of new test intervals; old test checked separately. Original frozen model also compared.\nNew intervals are from the same two held-out matches, not new independent matches.',RED)
    im.save(OUT/'second-protocol.png')

    result=json.loads((ROOT/'runs/tennis_second/comparison.json').read_text(encoding='utf-8'))
    im,d=canvas('04  Improvement, with substantial failures remaining', 'Fresh test: 5 clips / 28 seconds / 30 teacher frame anchors. Matching tolerance: +/-3.',1050)
    for x,name,label in [(40,'original','Original'),(490,'control','Calibration-only control'),(940,'revised','Revised loss')]:
        m=result['fresh_test'][name]['model']['3']
        box(d,(x,140,420,165),label,f"F1 {m['f1']:.3f} | precision {m['precision']:.1%}\nRecall {m['recall']:.1%}: {m['matched']}/30 anchors",GREEN if name=='revised' else BLUE)
    registry=json.loads((ROOT/'examples/tennis_second_teacher.json').read_text(encoding='utf-8'))
    for y,name,label in [(355,'second_gauff_rally','Gauff rally'),(545,'second_rybakina_rally','Rybakina rally'),(735,'second_gauff_waiting','Waiting players')]:
        clip=json.loads((ROOT/'data/tennis_second/clips'/name/'clip.json').read_text(encoding='utf-8'))
        prediction=json.loads((ROOT/'runs/tennis_second/fresh_test/revised/predictions'/(name+'.json')).read_text(encoding='utf-8'))
        truth={i for e in registry['clips'][name] for i in e['frame_indices']}
        predicted={i for e in prediction['event_sequences'] for i in e}
        d.text((40,y),label,font=font(25,True),fill=INK)
        for yy,caption,indices,color in [(y+48,'Teacher',truth,GREEN),(y+94,'Revised',predicted,RED)]:
            d.text((40,yy-5),caption,font=font(22),fill=MUTED)
            for i in range(len(clip['frames'])):
                x=320+i*14
                d.rectangle((x,yy,x+10,yy+25),fill=color if i in indices else '#dfe6ed')
        for tick in range(0,len(clip['frames']),10):
            d.text((320+tick*14,y+133),str(tick),font=font(19),fill=MUTED)
        d.text((40,y+133),'Filtered frame index',font=font(18),fill=MUTED)
    lines(d,'Green = teacher anchor. Red = model event-frame prediction (not automatically wrong).\nThe waiting clip has only false positives; the entire Rybakina rally was missed. Boundary context is omitted.',(40,940),1310,22,MUTED)
    im.save(OUT/'second-results.png')

    destination=ROOT/'docs/second-results';destination.mkdir(exist_ok=True)
    for name in ('comparison.json','diagnostics.json','protocol.json'):
        shutil.copyfile(ROOT/'runs/tennis_second'/name,destination/name)
    html=['<!doctype html><html lang="en"><meta charset="utf-8"><title>Tennis experiment 2</title>',
          '<style>body{font-family:Segoe UI,Arial,sans-serif;max-width:1400px;margin:32px auto;padding:0 24px;background:#f5f8fc;color:#15283e}img{width:100%;height:auto;margin:16px 0}p{font-size:20px;line-height:1.5}a{color:#1769aa}</style>',
          '<h1>Tennis selector: why the first pilot failed, and the second experiment</h1>',
          '<p>The revised model improved fresh-test F1 from 0.146 to 0.256, but recovered only 5 of 30 teacher anchors. It remains a research baseline.</p>',
          '<p><a href="second-experiment.md">Detailed explanation, measurements, limitations and reproduction steps</a></p>']
    for name in ('first-pilot','loss-conflict','second-protocol','second-results'):
        html.append(f'<img src="diagrams/{name}.png" alt="{name}">')
    html.append('</html>')
    (ROOT/'docs/second-experiment.html').write_text('\n'.join(html),encoding='utf-8')


if __name__=='__main__':main()
