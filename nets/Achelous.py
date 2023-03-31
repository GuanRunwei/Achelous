import torch
import torch.nn as nn
import math
import time
from thop import profile
from thop import clever_format
from torchinfo import summary
from backbone.vision.ImageEncoder import *
from neck.spp import *
from backbone.conv_utils.normal_conv import *
from backbone.conv_utils.ghost_conv import *
from backbone.IREncoder import IREncoder
from backbone.attention_modules.shuffle_attention import *
from head.decouplehead import DecoupleHead
from nets.pointcloudseg.pointnet2.pointnet_sem_seg import PointNet_SEG


image_encoder_width = {
    'L': [40, 80, 192, 384],  # 26m 83.3% 6attn
    'S2': [32, 64, 144, 288],  # 12m 81.6% 4attn dp0.02
    'S1': [32, 48, 120, 224],  # 6.1m 79.0
    'S0': [32, 48, 96, 176],  # 75.0 75.7
}


class Achelous(nn.Module):
    def __init__(self, num_det, num_seg, phi='SO', image_channels=3, radar_channels=3, resolution=416,
                 backbone='ef', neck='gdf', pc_seg='pn', pc_channels=6, pc_classes=9, nano_head=False):
        super(Achelous, self).__init__()

        if pc_seg == 'pn':
            self.pc_seg_model = PointNet_SEG(num_class=pc_classes, point_cloud_channels=pc_channels)

        self.num_det = num_det
        self.num_seg = num_seg
        self.resolution = resolution

        self.phi = phi
        self.image_channels = image_channels
        self.radar_channels = radar_channels

        self.image_radar_encoder = IREncoder(num_class_seg=num_seg, resolution=resolution, backbone=backbone, neck=neck,
                                             phi=phi)
        self.det_head = DecoupleHead(num_classes=num_det, phi=phi, nano_head=nano_head)

    def forward(self, x, x_radar, x_point_clouds):
        pc_seg_output = self.pc_seg_model(x_point_clouds)
        fpn_out, se_seg_output, lane_seg_output = self.image_radar_encoder(x, x_radar)
        det_output = self.det_head(fpn_out)
        return det_output, se_seg_output, lane_seg_output, pc_seg_output


class Achelous3T(nn.Module):
    def __init__(self, num_det, num_seg, phi='SO', image_channels=3, radar_channels=3, resolution=320,
                 backbone='en', neck='gdf', pc_seg='pn', pc_channels=6, pc_classes=9, nano_head=True):
        super(Achelous3T, self).__init__()

        self.num_det = num_det
        self.num_seg = num_seg
        self.resolution = resolution

        self.phi = phi
        self.image_channels = image_channels
        self.radar_channels = radar_channels

        self.image_radar_encoder = IREncoder(num_class_seg=num_seg, resolution=resolution, backbone=backbone, neck=neck,
                                             phi=phi)
        self.det_head = DecoupleHead(num_classes=num_det, phi=phi, nano_head=nano_head)

    def forward(self, x, x_radar):
        fpn_out, se_seg_output, lane_seg_output = self.image_radar_encoder(x, x_radar)
        det_output = self.det_head(fpn_out)
        return det_output, se_seg_output, lane_seg_output


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_map = torch.randn((1, 3, 320, 320)).to(device)
    input_map_radar = torch.randn((1, 3, 320, 320)).to(device)
    input_pc_radar = torch.randn((1, 6, 512)).to(device)
    model = Achelous(num_det=8, num_seg=9, phi='S0', resolution=320, backbone='ev', neck='gdf', pc_channels=6,
                     pc_classes=9).to(device)
    model.eval()
    output_map1, output_map2, output_map3, output_map4 = model(input_map, input_map_radar, input_pc_radar)
    print(output_map1[0].shape)
    print(output_map1[1].shape)
    print(output_map1[2].shape)
    print(output_map2.shape)
    print(output_map3.shape)
    print(output_map4[0].shape)

    print(summary(model, input_size=[(1, 3, 320, 320), (1, 3, 320, 320), (1, 6, 512)]))
    macs, params = profile(model, inputs=[input_map, input_map_radar, input_pc_radar])
    macs *= 2
    macs, params = clever_format([macs, params], "%.3f")
    print("FLOPs:", macs)
    print("Params:", params)

    t1 = time.time()
    test_times = 300
    for i in range(test_times):
        output = model(input_map, input_map_radar, input_pc_radar)
    t2 = time.time()
    print("fps:", (1 / ((t2 - t1) / test_times)))
