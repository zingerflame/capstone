"""Frozen ImageNet ResNet-18 features and a ten-frame temporal selector.

Architecture/weight key compatibility follows the published ResNet-18 topology.
Official weights: https://download.pytorch.org/models/resnet18-f37072fd.pth
This compact implementation uses PyTorch only, without torchvision binary ops.
"""
import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from .io import fingerprint, read_json, write_json
from .core import valid_windows

WEIGHTS_SHA256 = 'f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'


class ResidualBlock(nn.Module):
    def __init__(self, incoming, outgoing, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(incoming,outgoing,3,stride,1,bias=False)
        self.bn1 = nn.BatchNorm2d(outgoing)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outgoing,outgoing,3,1,1,bias=False)
        self.bn2 = nn.BatchNorm2d(outgoing)
        self.downsample = (nn.Sequential(nn.Conv2d(incoming,outgoing,1,stride,bias=False),
                                        nn.BatchNorm2d(outgoing))
                           if stride != 1 or incoming != outgoing else None)

    def forward(self, x):
        skip = x if self.downsample is None else self.downsample(x)
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))+skip)


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3,64,7,2,3,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU(inplace=True)
        self.maxpool=nn.MaxPool2d(3,2,1)
        self.layer1=nn.Sequential(ResidualBlock(64,64),ResidualBlock(64,64))
        self.layer2=nn.Sequential(ResidualBlock(64,128,2),ResidualBlock(128,128))
        self.layer3=nn.Sequential(ResidualBlock(128,256,2),ResidualBlock(256,256))
        self.layer4=nn.Sequential(ResidualBlock(256,512,2),ResidualBlock(512,512))
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Linear(512,1000)  # Load all official state keys strictly; unused for features.

    def forward(self,x):
        x=self.maxpool(self.relu(self.bn1(self.conv1(x))))
        return self.avgpool(self.layer4(self.layer3(self.layer2(self.layer1(x))))).flatten(1)


def load_encoder(weights):
    if hashlib.sha256(Path(weights).read_bytes()).hexdigest() != WEIGHTS_SHA256:
        raise ValueError('Unexpected pretrained weight checksum')
    model=ResNet18()
    model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
    model.eval().requires_grad_(False)
    return model


def full_frame_tensor(image):
    """224-square letterbox retains the full broadcast; differs from center-crop ImageNet recipe."""
    h,w=image.shape[:2]
    scale=224/max(h,w)
    resized=cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),
                       (max(1,round(w*scale)),max(1,round(h*scale))),interpolation=cv2.INTER_AREA)
    canvas=np.full((224,224,3),[124,116,104],dtype=np.uint8)
    hh,ww=resized.shape[:2]
    canvas[(224-hh)//2:(224-hh)//2+hh,(224-ww)//2:(224-ww)//2+ww]=resized
    value=torch.from_numpy(canvas.transpose(2,0,1).copy()).float()/255
    return (value-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]


def extract_features(directory,clip,encoder,cache,batch_size=32):
    """No labels or split statistics are used by feature extraction."""
    directory,cache=Path(directory),Path(cache)
    digest=hashlib.sha256()
    for row in clip['frames']:
        for field in ('image','scoreboard'):
            digest.update((directory/row[field]).read_bytes())
    key=fingerprint({'clip':clip,'images_sha256':digest.hexdigest(),'weights':WEIGHTS_SHA256,
                     'recipe':'letterbox224_imagenet_norm_board_gray16x8_v1'})
    cache.mkdir(parents=True,exist_ok=True)
    path=cache/(key+'.npy')
    if path.exists():
        return torch.from_numpy(np.load(path,allow_pickle=False))
    features=[]
    with torch.no_grad():
        for start in range(0,len(clip['frames']),batch_size):
            rows=clip['frames'][start:start+batch_size]
            images=[];boards=[];metadata=[]
            for row in rows:
                i=row['index']
                images.append(full_frame_tensor(cv2.imread(str(directory/row['image']))))
                board=cv2.imread(str(directory/row['scoreboard']),cv2.IMREAD_GRAYSCALE)
                present=float(clip.get('scoreboard_roi') is not None)
                boards.append(cv2.resize(board,(16,8)).reshape(-1).astype(np.float32)/255*present)
                dt=0 if i==0 else min(2,row['timestamp']-clip['frames'][i-1]['timestamp'])
                metadata.append([present,dt])
            scene=encoder(torch.stack(images)).cpu()
            features.append(torch.cat([scene,torch.tensor(np.array(boards)),torch.tensor(metadata)],dim=1))
    result=torch.cat(features).float()
    np.save(path,result.numpy())
    write_json(path.with_suffix('.json'),{'clip':clip['id'],'fingerprint':key,'shape':list(result.shape),
                                        'weights_sha256':WEIGHTS_SHA256})
    return result


class FeatureSelector(nn.Module):
    """Appearance-only and explicit-difference variants both operate strictly inside ten frames."""
    def __init__(self,motion=False,width=64):
        super().__init__()
        self.motion,self.width=motion,width
        self.scene_norm=nn.LayerNorm(512)
        self.fuse=nn.Linear(642,width)
        self.position=nn.Parameter(torch.zeros(1,10,width))
        layer=nn.TransformerEncoderLayer(width,4,width*2,dropout=.1,batch_first=True,norm_first=True)
        self.temporal=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.head=nn.Linear(width,1)
        # Initialize shared modules identically in both arms for the same seed.
        self.motion_projection=nn.Linear(512,width,bias=False) if motion else None

    def forward(self,x,valid):
        scene=self.scene_norm(x[:,:,:512])
        features=torch.cat([scene,x[:,:,512:640]*x[:,:,640:641],x[:,:,640:]],dim=-1)
        hidden=self.fuse(features)
        if self.motion_projection is not None:
            difference=torch.zeros_like(scene)
            difference[:,1:]=(scene[:,1:]-scene[:,:-1])*(valid[:,1:] & valid[:,:-1])[:,:,None]
            hidden=hidden+self.motion_projection(difference)
        hidden=hidden.masked_fill(~valid[:,:,None],0)
        return self.head(self.temporal(hidden+self.position,src_key_padding_mask=~valid)).squeeze(-1)


def feature_logits(model,features):
    ids,valid=map(torch.tensor,valid_windows(len(features)))
    return model(features[ids],valid),ids,valid
