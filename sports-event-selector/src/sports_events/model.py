import torch
from torch import nn


class ImageEncoder(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.GroupNorm(4, 16), nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GroupNorm(4, 32), nn.GELU(),
            nn.Conv2d(32, width, 3, stride=2, padding=1), nn.GroupNorm(8, width), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, x):
        return self.layers(x)


class Selector(nn.Module):
    """Full-frame + scoreboard CNNs -> temporal attention -> ten binary logits.

    This is multi-label frame selection, not mutually exclusive sport classification.
    No text, event class, VLM confidence or annotation enters the forward pass.
    """
    def __init__(self, width=64):
        super().__init__()
        self.width = width
        self.scene = ImageEncoder(width)
        self.board = ImageEncoder(width)
        self.fuse = nn.Linear(width * 2 + 2, width)
        self.position = nn.Parameter(torch.zeros(1, 10, width))
        layer = nn.TransformerEncoderLayer(width, 4, width * 2, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.head = nn.Linear(width, 1)

    def encode(self, frames, boards, board_present, time_delta):
        scene = self.scene(frames)
        board = self.board(boards) * board_present[:, None]
        return self.fuse(torch.cat([scene, board, board_present[:, None], time_delta[:, None]], dim=-1))

    def classify(self, embeddings, valid):
        hidden = self.temporal(embeddings + self.position, src_key_padding_mask=~valid)
        return self.head(hidden).squeeze(-1)
