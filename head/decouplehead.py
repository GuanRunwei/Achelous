import torch
import torch.nn as nn

from backbone.conv_utils.normal_conv import BaseConv



image_encoder_width = {
    'L': [40, 80, 192, 384],  # 26m 83.3% 6attn
    'S2': [32, 64, 144, 288],  # 12m 81.6% 4attn dp0.02
    'S1': [32, 48, 120, 224],  # 6.1m 79.0
    'S0': [32, 48, 96, 176],  # 75.0 75.7
}


class DecoupleHead(nn.Module):
    def __init__(self, num_classes, width=1.0, phi='S0', act="relu", depthwise=True, nano_head=True):
        super().__init__()
        Conv = BaseConv

        in_channels = [channels*5//4 for channels in image_encoder_width[phi][1:]]

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()

        if nano_head is True:
            base_num = 96
        else:
            base_num = 256

        for i in range(len(in_channels)):
            self.stems.append(
                Conv(in_channels=int(in_channels[i] * width), out_channels=int(base_num * width), ksize=1, stride=1,
                         act=act))
            self.cls_convs.append(nn.Sequential(*[
                Conv(in_channels=int(base_num * width), out_channels=int(base_num * width), ksize=5, stride=1, act=act, ds_conv=depthwise),
                Conv(in_channels=int(base_num * width), out_channels=int(base_num * width), ksize=5, stride=1, act=act, ds_conv=depthwise),
            ]))
            self.cls_preds.append(
                nn.Conv2d(in_channels=int(base_num * width), out_channels=num_classes, kernel_size=1, stride=1, padding=0)
            )

            self.reg_convs.append(nn.Sequential(*[
                Conv(in_channels=int(base_num * width), out_channels=int(base_num * width), ksize=5, stride=1, act=act, ds_conv=depthwise),
                Conv(in_channels=int(base_num * width), out_channels=int(base_num * width), ksize=5, stride=1, act=act, ds_conv=depthwise)
            ]))
            self.reg_preds.append(
                nn.Conv2d(in_channels=int(base_num * width), out_channels=4, kernel_size=1, stride=1, padding=0)
            )
            self.obj_preds.append(
                nn.Conv2d(in_channels=int(base_num * width), out_channels=1, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, inputs):
        # ---------------------------------------------------#
        #   inputs输入
        #   P3_out  80, 80, 256
        #   P4_out  40, 40, 512
        #   P5_out  20, 20, 1024
        # ---------------------------------------------------#
        outputs = []
        for k, x in enumerate(inputs):
            # ---------------------------------------------------#
            #   利用1x1卷积进行通道整合
            # ---------------------------------------------------#
            x = self.stems[k](x)
            # ---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            # ---------------------------------------------------#
            cls_feat = self.cls_convs[k](x)
            # ---------------------------------------------------#
            #   判断特征点所属的种类
            #   80, 80, num_classes
            #   40, 40, num_classes
            #   20, 20, num_classes
            # ---------------------------------------------------#
            cls_output = self.cls_preds[k](cls_feat)

            # ---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            # ---------------------------------------------------#
            reg_feat = self.reg_convs[k](x)
            # ---------------------------------------------------#
            #   特征点的回归系数
            #   reg_pred 80, 80, 4
            #   reg_pred 40, 40, 4
            #   reg_pred 20, 20, 4
            # ---------------------------------------------------#
            reg_output = self.reg_preds[k](reg_feat)
            # ---------------------------------------------------#
            #   判断特征点是否有对应的物体
            #   obj_pred 80, 80, 1
            #   obj_pred 40, 40, 1
            #   obj_pred 20, 20, 1
            # ---------------------------------------------------#
            obj_output = self.obj_preds[k](reg_feat)

            output = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)
        return outputs