"""Download publisher-linked public videos, sequentially and atomically."""
import hashlib, json, time, urllib.request, urllib.parse
from html.parser import HTMLParser
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/'data/tenniset'
VIDEOS={'V006':('1w1R6R6BAa_9K-he5nqygT5Bvb5CsyKhT',1691942447),
        'V007':('12M3X2nxkOrjjUhXyyDxB42CBy0Mvj937',1753252295),
        'V008':('1cIRteQttU5YTSMaa-_icXv-eqt2Fo4dH',995156723),
        'V009':('15-juiUBq29FYsfUTpsmYaMXNz3GxvTaQ',5316283504),
        'V010':('169cmHfVdpikwGdS3xnsq0t_LQBSYjCBu',1333929688)}
class Form(HTMLParser):
    def __init__(self): super().__init__();self.action=None;self.fields={}
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='form' and a.get('id')=='download-form':self.action=a['action']
        if tag=='input' and a.get('type')=='hidden':self.fields[a['name']]=a.get('value','')
ROOT.joinpath('videos').mkdir(parents=True,exist_ok=True)
for name,(file_id,size) in VIDEOS.items():
    target=ROOT/'videos'/f'{name}.mp4'
    if target.exists() and target.stat().st_size==size:continue
    url=f'https://drive.google.com/uc?export=download&id={file_id}'
    response=urllib.request.urlopen(url,timeout=120)
    if 'text/html' in response.headers.get('Content-Type',''):
        form=Form();form.feed(response.read().decode());response.close()
        if form.action!='https://drive.usercontent.google.com/download':raise ValueError('Unexpected download form')
        response=urllib.request.urlopen(form.action+'?'+urllib.parse.urlencode(form.fields),timeout=120)
    if 'text/html' in response.headers.get('Content-Type',''):raise ValueError('Video unavailable')
    partial=target.with_suffix('.partial');digest=hashlib.sha256();count=0;last=time.monotonic()
    with response,partial.open('wb') as stream:
        while block:=response.read(4*1024*1024):
            stream.write(block);digest.update(block);count+=len(block)
            if time.monotonic()-last>15:print(f'{name}: {count/size:.1%}',flush=True);last=time.monotonic()
    if count!=size:raise ValueError(f'{name}: size mismatch {count} != {size}')
    partial.replace(target)
    target.with_suffix('.source.json').write_text(json.dumps({'video_id':name,'drive_id':file_id,'bytes':count,'sha256':digest.hexdigest(),'url':url},indent=2))
    print(f'{name} complete ({count} bytes)',flush=True)

