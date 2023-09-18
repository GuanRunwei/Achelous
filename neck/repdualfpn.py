import torch
import torch.nn as nn
import math
import time
from thop import profile
from thop import clever_format
from torchinfo import summary
from backbone.conv_utils.repvgg_conv import RepVGGBlock
from backbone.vision.ImageEncoder import *
from neck.spp import *
from backbone.conv_utils.normal_conv import *
from backbone.conv_utils.ghost_conv import *
from neck.sobel import Sobel
from backbone.attention_modules.shuffle_attention import *
from backbone.vision.fastvit_modules.fastvit import *
from backbone.vision.mobileone_modules.mobileone import *


image_encoder_width = {
    'L': [40, 80, 192, 384],  # 26m 83.3% 6attn
    'S2': [32, 64, 144, 288],  # 12m 81.6% 4attn dp0.02
    'S1': [32, 48, 120, 224],  # 6.1m 79.0
    'S0': [32, 48, 96, 176],  # 75.0 75.7
}


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2, ds_conv=False):
        super().__init__()

        self.upsample = nn.Sequential(
            BaseConv(in_channels, out_channels, 1, 1, act='relu', ds_conv=ds_conv),
            nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        x = self.upsample(x)
        return x


class RepDualFPN(nn.Module):
    def __init__(self, num_class_seg, resolution=320, phi='S0', use_spp=True, backbone='mo'):
        super().__init__()

        self.phi = phi
        self.channel_widths = image_encoder_width[phi]
        self.use_spp = use_spp
        self.resolution = resolution
        self.num_class_seg = num_class_seg
        if self.num_class_seg > 31:
            assert "class number of semantic segmentation must be smaller than 32 (<=31)"

        if phi == 'S0':
            if backbone == 'mo':
                self.backbone = mobileone(variant='S0')
            elif backbone == 'fv':
                self.backbone = fastvit_s0(fork_feat=True)

        elif phi == 'S1':
            if backbone == 'mo':
                self.backbone = mobileone(variant='S1')
            elif backbone == 'fv':
                self.backbone = fastvit_s1(fork_feat=True)

        elif phi == 'S2':
            if backbone == 'mo':
                self.backbone = mobileone(variant='S2')
            elif backbone == 'fv':
                self.backbone = fastvit_s2(fork_feat=True)

        if use_spp:
            self.spp = SPP(c1=self.channel_widths[-1], c2=self.channel_widths[-1])
        else:
            self.spp = SPPF(c1=self.channel_widths[-1], c2=self.channel_widths[-1])

        # 176, 16, 16 -> 192, 32, 32
        self.upsample_5_to_4 = Upsample(in_channels=self.channel_widths[-1], out_channels=self.channel_widths[-2])
        self.ghost_5_to_4 = RepVGGBlock(in_channels=self.channel_widths[-2]*2, out_channels=self.channel_widths[-2],
                                        kernel_size=3, padding=1)

        # 192, 32, 32 -> 96, 64, 64
        self.upsample_4_to_3 = Upsample(in_channels=self.channel_widths[-2], out_channels=self.channel_widths[-3])
        self.ghost_4_to_3 = RepVGGBlock(in_channels=self.channel_widths[-3]*2, out_channels=self.channel_widths[-3],
                                        kernel_size=3, padding=1)

        # 96, 64, 64 -> lane-segmentation 96, 64, 64
        self.stage_3_lane_seg = ShuffleAttention(channel=self.channel_widths[-3], G=4)
        # 48, 64, 64 -> semantic-segmentation 96, 64, 64
        self.stage_3_semantic_seg = ShuffleAttention(channel=self.channel_widths[-3], G=4)

        # ======================================= WaterLine Segmentation ========================================= #
        # lane-segmentation 96, 64, 64 -> 48, 128, 128
        self.lane_seg_3_to_2 = Upsample(in_channels=self.channel_widths[-3], out_channels=self.channel_widths[-3])
        self.lane_seg_ghost_3_to_2 = RepVGGBlock(in_channels=self.channel_widths[-3], out_channels=self.channel_widths[-3],
                                                 kernel_size=3, padding=1)

        # lane-segmentation 48, 128, 128 -> 32, 256, 256
        self.lane_seg_2_to_1 = Upsample(in_channels=self.channel_widths[-3], out_channels=self.channel_widths[-4])
        self.lane_seg_ghost_2_to_1 = RepVGGBlock(in_channels=self.channel_widths[-4], out_channels=self.channel_widths[-4],
                                                 kernel_size=3, padding=1)

        # lane-segmentation 32, 256, 256 -> 32, 512, 512
        self.lane_seg_1_to_0 = Upsample(in_channels=self.channel_widths[-4], out_channels=self.channel_widths[-4])
        self.lane_seg_ghost_1_to_0 = RepVGGBlock(in_channels=self.channel_widths[-4], out_channels=self.channel_widths[-4],
                                                 kernel_size=3, padding=1)

        # lane-segmentation 32, 512, 512 -> 2, 512, 512
        self.lane_seg_head = RepVGGBlock(in_channels=self.channel_widths[-4], out_channels=2,
                                                 kernel_size=3, padding=1)
        # ======================================================================================================== #

        # ======================================= Semantic Segmentation ========================================== #
        # semantic-segmentation 96, 64, 64 -> 48, 128, 128
        self.se_seg_3_to_2 = Upsample(in_channels=self.channel_widths[-3], out_channels=self.channel_widths[-3])
        self.se_seg_ghost_3_to_2 = RepVGGBlock(in_channels=self.channel_widths[-3], out_channels=self.channel_widths[-3],
                                                 kernel_size=3, padding=1)

        # semantic-segmentation 48, 128, 128 -> 32, 256, 256
        self.se_seg_2_to_1 = Upsample(in_channels=self.channel_widths[-3], out_channels=self.channel_widths[-4])
        self.se_seg_ghost_2_to_1 = RepVGGBlock(in_channels=self.channel_widths[-4], out_channels=self.channel_widths[-4],
                                                 kernel_size=3, padding=1)

        # semantic-segmentation 32, 256, 256 -> 32, 512, 512
        self.se_seg_1_to_0 = Upsample(in_channels=self.channel_widths[-4], out_channels=self.channel_widths[-4])
        self.se_seg_ghost_1_to_0 = RepVGGBlock(in_channels=self.channel_widths[-4], out_channels=self.channel_widths[-4],
                                                 kernel_size=3, padding=1)

        # semantic-segmentation 32, 512, 512 -> num_class_seg, 512, 512
        self.se_seg_head = RepVGGBlock(in_channels=self.channel_widths[-4], out_channels=self.num_class_seg,
                                       kernel_size=3, padding=1)
        # ======================================================================================================== #

        # self.fpn_stage5_det = GhostModule(inp=self.channel_widths[-1], oup=self.channel_widths[-1]*2)

    def forward(self, x):
        map_stage2, map_stage3, map_stage4, map_stage5 = self.backbone(x)

        # fpn_stage5_det = self.fpn_stage5_det(map_stage5)

        fpn_stage5 = self.spp(map_stage5)

        fpn_stage4 = self.upsample_5_to_4(fpn_stage5)
        fpn_stage4 = torch.cat([fpn_stage4, map_stage4], dim=1)
        fpn_stage4 = self.ghost_5_to_4(fpn_stage4)

        fpn_stage3 = self.upsample_4_to_3(fpn_stage4)
        fpn_stage3 = torch.cat([fpn_stage3, map_stage3], dim=1)
        fpn_stage3 = self.ghost_4_to_3(fpn_stage3)

        fpn_stage3_lane = self.stage_3_lane_seg(fpn_stage3)
        fpn_stage3_semantic = self.stage_3_semantic_seg(fpn_stage3)

        # ============================= lane seg ============================== #
        fpn_stage2_lane = self.lane_seg_3_to_2(fpn_stage3_lane)
        fpn_stage2_lane = self.lane_seg_ghost_3_to_2(fpn_stage2_lane)

        fpn_stage1_lane = self.lane_seg_2_to_1(fpn_stage2_lane)
        fpn_stage1_lane = self.lane_seg_ghost_2_to_1(fpn_stage1_lane)

        fpn_stage0_lane = self.lane_seg_1_to_0(fpn_stage1_lane)
        fpn_stage0_lane = self.lane_seg_ghost_1_to_0(fpn_stage0_lane)

        lane_seg_output = self.lane_seg_head(fpn_stage0_lane)
        # ===================================================================== #

        # ========================= semantic segmentation ===================== #
        fpn_stage2_se = self.se_seg_3_to_2(fpn_stage3_semantic)
        fpn_stage2_se = self.se_seg_ghost_3_to_2(fpn_stage2_se)

        fpn_stage1_se = self.se_seg_2_to_1(fpn_stage2_se)
        fpn_stage1_se = self.se_seg_ghost_2_to_1(fpn_stage1_se)

        fpn_stage0_se = self.se_seg_1_to_0(fpn_stage1_se)
        fpn_stage0_se = self.se_seg_ghost_1_to_0(fpn_stage0_se)

        se_seg_output = self.se_seg_head(fpn_stage0_se)
        # ===================================================================== #

        return se_seg_output, lane_seg_output, (fpn_stage5+map_stage5, fpn_stage4+map_stage4, fpn_stage3+map_stage3)


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    input_map = torch.randn((1, 3, 416, 416)).to(device)
    model = RepDualFPN(num_class_seg=9, phi='S1', resolution=416).to(device)
    model.eval()

    for child in model.children():
        if isinstance(child, nn.Module):
            print('deploy:', type(child).__name__)
            child.deploy = True

    for module in model.modules():
        if hasattr(module, 'reparameterize'):
            print('reparameterize:', type(module).__name__)
            module.reparameterize()

    output_map1, output_map2, _= model(input_map)
    print(output_map1.shape)
    print(output_map2.shape)

    print(summary(model, input_size=(1, 3, 416, 416)))
    macs, params = profile(model, inputs=input_map.unsqueeze(0))
    macs, params = clever_format([macs, params], "%.3f")

    t1 = time.time()
    test_times = 300
    for i in range(test_times):
        output = model(input_map)
    t2 = time.time()
    print("fps:", (1 / ((t2 - t1) / test_times)))


