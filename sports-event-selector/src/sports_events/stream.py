"""Incremental inference for an already delta-filtered BGR frame feed."""
from collections import deque
import cv2
import numpy as np
import torch
from .core import boundaries, decode
from .video import crop_scoreboard


class FrameStream:
    """Keep ten visual embeddings; defer final global budget until finish().

    push() returns window probabilities, not irrevocable event decisions. Exact
    global budget ranking and mandatory last-two retention require end-of-stream.
    Raw frames are not retained. Numeric window predictions use O(N) memory.
    """
    def __init__(self, model, budget_rate, scoreboard_roi=None, device="cpu", threshold=.5, consensus=.5):
        self.model = model.to(device).eval()
        self.rate, self.roi, self.device = budget_rate, scoreboard_roi, device
        self.threshold, self.consensus = threshold, consensus
        self.features = deque(maxlen=10)
        self.probabilities, self.timestamps = [], []
        self.closed = False

    def push(self, bgr_frame, timestamp):
        if self.closed:
            raise RuntimeError("Stream already finished")
        if not np.isfinite(timestamp) or (self.timestamps and timestamp <= self.timestamps[-1]):
            raise ValueError("Feed timestamps must be finite and strictly increasing")
        def tensor(image, size):
            rgb = cv2.cvtColor(cv2.resize(image,size), cv2.COLOR_BGR2RGB)
            return torch.from_numpy(rgb.transpose(2,0,1).copy()).float().unsqueeze(0).to(self.device)/255
        dt = 0 if not self.timestamps else min(2, timestamp-self.timestamps[-1])
        with torch.no_grad():
            encoded = self.model.encode(tensor(bgr_frame,(96,96)), tensor(crop_scoreboard(bgr_frame,self.roi),(96,32)),
                torch.tensor([float(self.roi is not None)],device=self.device),
                torch.tensor([dt],device=self.device,dtype=torch.float32))
            self.features.append(encoded[0])
            self.timestamps.append(float(timestamp))
            if len(self.features) == 10:
                p = self.model.classify(torch.stack(list(self.features))[None],
                     torch.ones((1,10),dtype=torch.bool,device=self.device)).sigmoid()[0].cpu().numpy()
                self.probabilities.append(p)
                return {"window_start":len(self.timestamps)-10, "probabilities":p.tolist()}
        return None

    def finish(self):
        n = len(self.timestamps)
        if not n:
            raise ValueError("Cannot finish an empty stream")
        if n < 10 and not self.probabilities:
            features = list(self.features)
            features.extend([features[-1]]*(10-n))
            valid = torch.arange(10,device=self.device)[None] < n
            with torch.no_grad():
                self.probabilities.append(self.model.classify(torch.stack(features)[None],valid).sigmoid()[0].cpu().numpy())
        self.closed = True
        budget = len(boundaries(n)) + round(self.rate*max(0,n-len(boundaries(n))))
        result = decode(np.stack(self.probabilities),n,budget,self.threshold,self.consensus)
        result["sequence_timestamps"] = [[self.timestamps[i] for i in seq] for seq in result["event_sequences"]]
        return result
