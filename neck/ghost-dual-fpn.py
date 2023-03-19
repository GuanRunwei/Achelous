import torch
import torch.nn as nn
import math
from backbone.vision.ImageEncoder import *
from neck.spp import *


image_encoder_width = {
    'L': [40, 80, 192, 384],  # 26m 83.3% 6attn
    'S2': [32, 64, 144, 288],  # 12m 81.6% 4attn dp0.02
    'S1': [32, 48, 120, 224],  # 6.1m 79.0
    'S0': [32, 48, 96, 176],  # 75.0 75.7
}


class GhostDualFPN(nn.Module):
    def __init__(self, phi='S0', use_spp=True):
        super(GhostDualFPN, self).__init__()

        self.phi = phi
        self.channel_widths = image_encoder_width[phi]

        if phi == 'S0':
            self.backbone = image_encoder_s0()
        elif phi == 'S1':
            self.backbone = image_encoder_s1()
        elif phi == 'S2':
            self.backbone = image_encoder_s2()
        elif phi == 'L':
            self.backbone = image_encoder_l()

        if use_spp:
            self.spp = SPP(c1=self.channel_widths[-1], c2=self.channel_widths[-1])


