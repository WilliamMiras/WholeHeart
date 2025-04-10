import torch
import torch.nn as nn
from torchvision.models import convnext_tiny
from mamba_ssm import Mamba


class EchoNetConvMamba(nn.Module):
    def __init__(self):
        super().__init__()
        self.convnext = convnext_tiny(pretrained=True)
        self.convnext.classifier = nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.mamba = nn.Sequential(
            Mamba(d_model=768, d_state=16, d_conv=4, expand=2),
            Mamba(d_model=768, d_state=16, d_conv=4, expand=2)
        )

        self.ef_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.seg_head = nn.Sequential(
            nn.ConvTranspose2d(768, 384, 4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(384, 192, 4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(192, 96, 4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(96, 1, 3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, F, C, H, W = x.shape
        x = x.view(B * F, C, H, W)
        feats = self.convnext(x)
        feats = self.pool(feats)
        feats = feats.view(B, F, 768)

        temporal_feats = self.mamba(feats)

        ef = temporal_feats.mean(dim=1)
        ef = self.ef_head(ef) * 100

        seg = temporal_feats.view(B * F, 768, 1, 1)
        seg = self.seg_head(seg)
        seg = seg.view(B, F, 1, H, W)

        return ef, seg


def loss_fn(ef_pred, ef_true, seg_pred, seg_true):
    ef_loss = nn.MSELoss()(ef_pred, ef_true)
    seg_loss = nn.BCELoss()(seg_pred, seg_true)
    return ef_loss + 0.5 * seg_loss