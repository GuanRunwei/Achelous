import torch
import torch.nn as nn
import math
import time
from thop import profile
from thop import clever_format
from torchinfo import summary
from neck.spp import *
from backbone.conv_utils.normal_conv import *
from backbone.conv_utils.ghost_conv import *
from backbone.attention_modules.shuffle_attention import *
from backbone.conv_utils.dcn import DeformableConv2d


image_encoder_width = {
    'L': [40, 80, 192, 384],  # 26m 83.3% 6attn
    'S2': [32, 64, 144, 288],  # 12m 81.6% 4attn dp0.02
    'S1': [32, 48, 120, 224],  # 6.1m 79.0
    'S0': [32, 48, 96, 176],  # 75.0 75.7
}


class RadarConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation=2, is_dilation=False):
        super(RadarConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if is_dilation is True:
            self.initial_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                  stride=stride, padding=stride * 2, dilation=dilation)
        else:
            self.initial_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                          stride=stride, padding=kernel_size // 2)

        self.deformable_conv = DeformableConv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                                                stride=stride, padding=3 // 2)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.deformable_conv(x)
        return x


class RCBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=False):
        super(RCBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.radar_conv = RadarConv(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1)

        self.weight_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1,
                                      padding=0)

        self.norm = nn.BatchNorm2d(in_channels)
        self.activation = nn.ReLU(inplace=True)

        if down is False:
            self.weight_conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
                                          padding=0)
        else:
            self.weight_conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2,
                                          padding=1)

    def forward(self, x):
        x_res = x
        x = self.radar_conv(x)
        x = self.weight_conv1(x)
        x = self.norm(x)
        x = self.activation(x)
        x = x_res + x
        x = self.weight_conv2(x)

        return x


class RCNet(nn.Module):
    def __init__(self, in_channels, phi='S0'):
        super(RCNet, self).__init__()
        self.phi = phi
        self.in_channels = in_channels

        stage_blocks = []
        for i in range(4):
            if i == 0:
                stage_blocks.append(RCBlock(in_channels=in_channels,
                                            out_channels=image_encoder_width[phi][i] // 4, down=True))
                stage_blocks.append(RCBlock(in_channels=image_encoder_width[phi][i] // 4,
                                            out_channels=image_encoder_width[phi][i] // 4, down=True))
            else:
                stage_blocks.append(RCBlock(in_channels=image_encoder_width[phi][i-1] // 4,
                                            out_channels=image_encoder_width[phi][i-1] // 4, down=False))
                stage_blocks.append(RCBlock(in_channels=image_encoder_width[phi][i-1] // 4,
                                            out_channels=image_encoder_width[phi][i] // 4, down=True))

        self.rc_blocks = nn.ModuleList(stage_blocks)
        print(len(self.rc_blocks))

    def forward_blocks(self, x):
        output_features = []
        for i, block in enumerate(self.rc_blocks):
            x = block(x)
            if i > 1 and i % 2 == 1:
                output_features.append(x)
        return output_features

    def forward(self, x):
        x = self.forward_blocks(x)
        return x







if __name__ == '__main__':
    input_map = torch.randn(1, 3, 416, 416)
    model = RCNet(in_channels=3)
    output = model(input_map)
    print(len(output))
    print(output[0].shape)
    print(output[1].shape)
    print(output[2].shape)