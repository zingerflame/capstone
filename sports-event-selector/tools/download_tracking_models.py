"""Download official detector/ReID checkpoints with pinned SHA256 checksums."""
import argparse
from pathlib import Path
import urllib.request
from sports_events.broadcast_tracking import YOLOX_SHA,YOLOX_URL,OSNET_SHA
from sports_events.vertex_dataset import sha

OSNET_URL=('https://huggingface.co/kaiyangzhou/osnet/resolve/main/'
    'osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth')


def download(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    for name,url,digest in [('object_detection_yolox_2022nov.onnx',YOLOX_URL,YOLOX_SHA),
                            ('osnet_x0_25_msmt17.pth',OSNET_URL,OSNET_SHA)]:
        path=directory/name
        if not path.exists():
            temporary=path.with_suffix(path.suffix+'.download')
            urllib.request.urlretrieve(url,temporary)
            if sha(temporary)!=digest:raise ValueError('Downloaded weights failed SHA256 verification: '+name)
            temporary.replace(path)
        if sha(path)!=digest:raise ValueError('Existing checkpoint failed verification: '+name)
        print(name,'verified')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='data/pretrained')
    download(p.parse_args().output)
