"""Torch imports are optional: the baseline works without torch."""
import torch
from torch import nn
import torch.nn.functional as F

class CausalConv(nn.Module):
    def __init__(self,cin,cout,kernel=3,dilation=1):
        super().__init__(); self.left=(kernel-1)*dilation
        self.conv=nn.Conv1d(cin,cout,kernel,dilation=dilation)
    def forward(self,x): return self.conv(F.pad(x,(self.left,0)))

class Block(nn.Module):
    def __init__(self,cin,cout,dilation,dropout):
        super().__init__()
        self.a=CausalConv(cin,cout,dilation=dilation)
        self.b=CausalConv(cout,cout,dilation=dilation)
        # LayerNorm is per timestep; BatchNorm across time would leak future frames.
        self.na=nn.LayerNorm(cout); self.nb=nn.LayerNorm(cout)
        self.drop=nn.Dropout(dropout)
        self.skip=nn.Conv1d(cin,cout,1) if cin!=cout else nn.Identity()
    def norm(self,x,n): return n(x.transpose(1,2)).transpose(1,2)
    def forward(self,x):
        y=self.drop(F.relu(self.norm(self.a(x),self.na)))
        y=self.drop(F.relu(self.norm(self.b(y),self.nb)))
        return F.relu(y+self.skip(x))

class CropEncoder(nn.Module):
    """Small CNN trained on project labels; no hidden ImageNet download."""
    def __init__(self):
        super().__init__()
        layers=[]; cin=3
        for cout in (16,32,64):
            layers.extend([nn.Conv2d(cin,cout,3,2,1),nn.ReLU()]); cin=cout
        self.net=nn.Sequential(*layers,nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(64,128))
    def forward(self,x): return self.net(x)

class MotionTCN(nn.Module):
    def __init__(self,feature_dim,cfg):
        super().__init__(); self.use_rgb=cfg["use_rgb"]
        self.motion=nn.Sequential(nn.Linear(feature_dim,128),nn.ReLU(),nn.Linear(128,128))
        self.rgb=CropEncoder() if self.use_rgb else None
        channels=cfg["channels"]; cin=256 if self.use_rgb else 128
        self.tcn=nn.Sequential(*[Block(cin if i==0 else channels,channels,d,cfg["dropout"])
                                for i,d in enumerate(cfg["dilations"])])
        self.head=nn.Conv1d(channels,7,1)
        self.activity_head = None
        if cfg.get("activity_aux_weight"):
            with torch.random.fork_rng(devices=[]):
                self.activity_head = nn.Conv1d(channels, len(cfg.get("activity_aux_labels", ["bend", "sit", "lie"])), 1)

    def forward(self, x, rgb=None, return_activity=False):
        z=self.motion(x)
        if self.use_rgb:
            if rgb is None: raise ValueError("RGB checkpoint requires crops")
            b,t,c,h,w=rgb.shape
            image=self.rgb(rgb.reshape(b*t,c,h,w)).reshape(b,t,128)
            z=torch.cat([z,image],dim=-1)
        encoded = self.tcn(z.transpose(1, 2))
        logits = self.head(encoded).transpose(1, 2)
        if return_activity:
            return logits, self.activity_head(encoded).transpose(1, 2)
        return logits


class MotionCNNLSTM(nn.Module):
    """Causal 1D convolution followed by a forward LSTM on MotionGuard features."""

    def __init__(self, feature_dim, cfg):
        super().__init__()
        if cfg["use_rgb"]:
            raise ValueError("CNN-LSTM requires pose-only input")
        channels = cfg["channels"]
        self.conv = CausalConv(feature_dim, channels)
        self.lstm = nn.LSTM(channels, channels, batch_first=True)
        self.dropout = nn.Dropout(cfg["dropout"])
        self.head = nn.Linear(channels, 7)

    def forward(self, x, rgb=None):
        local = F.relu(self.conv(x.transpose(1, 2))).transpose(1, 2)
        sequence, _ = self.lstm(local)
        return self.head(self.dropout(sequence))


TEMPORAL_MODELS = {"tcn": MotionTCN, "cnn-lstm": MotionCNNLSTM}
class MotionSTGCN(nn.Module):
    """Learn a spatial graph on COCO joints, then apply causal temporal blocks."""

    def __init__(self, feature_dim, cfg):
        super().__init__()
        if feature_dim != 288 or cfg["use_rgb"]:
            raise ValueError("ST-GCN requires MotionGuard's 288-dimensional pose features")
        edges = [(0,1),(0,2),(1,3),(2,4),(0,5),(0,6),(5,6),(5,7),(7,9),
                 (6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
        adjacency = torch.eye(17)
        for a, b in edges:
            adjacency[a, b] = adjacency[b, a] = 1
        self.register_buffer("edge_mask", adjacency.bool())
        self.edge_logits = nn.Parameter(torch.zeros_like(adjacency))
        self.joint = nn.Linear(7, 16)
        self.spatial = nn.Linear(16, 16)
        channels = cfg["channels"]
        self.project = nn.Linear(feature_dim + 17 * 16, channels)
        self.temporal = nn.Sequential(
            *(Block(channels, channels, d, cfg["dropout"]) for d in cfg["dilations"]))
        self.head = nn.Conv1d(channels, 7, 1)

    def forward(self, x, rgb=None):
        b, t, _ = x.shape
        joints = torch.cat((x[..., :34].reshape(b,t,17,2),
                            x[..., 68:85].unsqueeze(-1),
                            x[..., 85:102].unsqueeze(-1),
                            x[..., 102:119].unsqueeze(-1),
                            x[..., 119:153].reshape(b,t,17,2)), dim=-1)
        weights = self.edge_logits.masked_fill(~self.edge_mask, -torch.inf).softmax(-1)
        embedded = F.relu(self.joint(joints))
        spatial = F.relu(self.spatial(weights @ embedded)) + embedded
        encoded = F.relu(self.project(torch.cat((x, spatial.flatten(2)), dim=-1)))
        return self.head(self.temporal(encoded.transpose(1,2))).transpose(1,2)


TEMPORAL_MODELS["stgcn"] = MotionSTGCN
