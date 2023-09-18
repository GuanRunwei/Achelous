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
from backbone.attention_modules.eca import eca_block
from backbone.radar.RadarEncoder import RCNet
from neck.ghostdualfpn import GhostDualFPN
from neck.cspdualfpn import CSPDualFPN
from neck.repdualfpn import RepDualFPN


image_encoder_width = {
    'L': [40, 80, 192, 384],  # 26m 83.3% 6attn
    'S2': [32, 64, 144, 288],  # 12m 81.6% 4attn dp0.02
    'S1': [32, 48, 120, 224],  # 6.1m 79.0
    'S0': [32, 48, 96, 176],  # 75.0 75.7
}


class IREncoder(nn.Module):
    def __init__(self, num_class_seg, phi='S0', resolution=416, use_spp=False, image_channels=3, radar_channels=3, backbone='ef', neck='gdf'):
        super(IREncoder, self).__init__()

        self.num_class_seg = num_class_seg
        self.phi = phi

        if neck == 'gdf':
            self.fpn = GhostDualFPN(num_class_seg=num_class_seg, phi=phi, resolution=resolution, use_spp=use_spp,
                                    backbone=backbone)
        elif neck == 'cdf':
            self.fpn = CSPDualFPN(num_class_seg=num_class_seg, phi=phi, resolution=resolution, use_spp=use_spp,
                                  backbone=backbone)
        elif neck == 'rdf':
            self.fpn = RepDualFPN(num_class_seg=num_class_seg, phi=phi, resolution=resolution, use_spp=use_spp,
                                  backbone=backbone)

        self.radar_encoder = RCNet(in_channels=radar_channels, phi=phi)

        # ================================= stage 3 ===================================== #
        # ------------------------ 分别度量每个分支内的通道重要程度 --------------------- #
        self.channel_attn_stage3 = nn.ModuleList([eca_block(channel=image_encoder_width[phi][1]) if i == 0
                                                  else eca_block(channel=image_encoder_width[phi][1] // 4)
                                                  for i in range(2)])
        # ------------------------------------------------------------------------------ #
        self.norm_stage3 = nn.BatchNorm2d(image_encoder_width[phi][1]*5//4)
        self.act_stage3 = nn.ReLU(inplace=True)
        # =============================================================================== #

        # ================================= stage 4 ===================================== #
        self.channel_attn_stage4 = nn.ModuleList([eca_block(channel=image_encoder_width[phi][2]) if i == 0
                                                  else eca_block(channel=image_encoder_width[phi][2] // 4)
                                                  for i in range(2)])
        self.norm_stage4 = nn.BatchNorm2d(image_encoder_width[phi][2]*5//4)
        self.act_stage4 = nn.ReLU(inplace=True)
        # =============================================================================== #

        # ================================= stage 5 ===================================== #
        self.channel_attn_stage5 = nn.ModuleList([eca_block(channel=image_encoder_width[phi][3]) if i == 0
                                                  else eca_block(channel=image_encoder_width[phi][3] // 4)
                                                  for i in range(2)])
        self.norm_stage5 = nn.BatchNorm2d(image_encoder_width[phi][3]*5//4)
        self.act_stage5 = nn.ReLU(inplace=True)
        # =============================================================================== #

    def forward(self, x, x_radar):
        image_features = self.fpn(x)
        se_seg_output, lane_seg_output, (map_stage5, map_stage4, map_stage3) = image_features

        radar_features = self.radar_encoder(x_radar)
        radar_stage3, radar_stage4, radar_stage5 = radar_features

        p3_ir_fuse = torch.cat([self.channel_attn_stage3[0](map_stage3), self.channel_attn_stage3[1](radar_stage3)],
                               dim=1)
        p3_fuse_out = self.act_stage3(self.norm_stage3(p3_ir_fuse))

        p4_ir_fuse = torch.cat([self.channel_attn_stage4[0](map_stage4), self.channel_attn_stage4[1](radar_stage4)],
                               dim=1)
        p4_fuse_out = self.act_stage4(self.norm_stage4(p4_ir_fuse))

        p5_ir_fuse = torch.cat([self.channel_attn_stage5[0](map_stage5), self.channel_attn_stage5[1](radar_stage5)],
                               dim=1)
        p5_fuse_out = self.act_stage5(self.norm_stage5(p5_ir_fuse))

        return (p3_fuse_out, p4_fuse_out, p5_fuse_out), se_seg_output, lane_seg_output


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    input_map = torch.randn((1, 3, 320, 320)).to(device)
    input_map_radar = torch.randn((1, 3, 320, 320)).to(device)
    model = IREncoder(num_class_seg=9, phi='S2', resolution=320).to(device)
    output_map1, output_map2, output_map3 = model(input_map, input_map_radar)
    print(output_map1[0].shape)
    print(output_map1[1].shape)
    print(output_map1[2].shape)
    print(output_map2.shape)
    print(output_map3.shape)

    print(summary(model, input_size=[(1, 3, 320, 320), (1, 3, 320, 320)]))
    macs, params = profile(model, inputs=[input_map, input_map_radar])
    macs *= 2
    macs, params = clever_format([macs, params], "%.3f")
    print("FLOPs:", macs)
    print("Params:", params)

    t1 = time.time()
    test_times = 300
    for i in range(test_times):
        output = model(input_map, input_map_radar)
    t2 = time.time()
    print("fps:", (1 / ((t2 - t1) / test_times)))


