import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from thop import profile
from thop import clever_format
from torchinfo import summary
import time
from nets.pointcloudseg.pointnet2.pointnet_utils import PointNetEncoder, feature_transform_reguliarzer


class PointNet_SEG(nn.Module):
    def __init__(self, num_class, point_cloud_channels):
        super(PointNet_SEG, self).__init__()
        self.k = num_class
        self.feat = PointNetEncoder(global_feat=False, feature_transform=True, channel=point_cloud_channels)
        self.conv1 = torch.nn.Conv1d(160, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 100, 1)
        self.conv3 = torch.nn.Conv1d(100, 64, 1)
        self.conv4 = torch.nn.Conv1d(64, self.k, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(100)
        self.bn3 = nn.BatchNorm1d(64)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x


class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat, weight):
        loss = F.nll_loss(pred, target, weight = weight)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)
        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss


if __name__ == '__main__':
    model = PointNet_SEG(num_class=9, point_cloud_channels=6).cuda()
    model.eval()
    xyz = torch.rand(1, 6, 512).cuda()

    output_map = model(xyz)
    print(output_map[0].shape)
    print(output_map[1].shape)

    macs, params = profile(model, inputs=xyz.unsqueeze(0))
    macs *= 2
    macs, params = clever_format([macs, params], "%.3f")
    print("FLOPs:", macs)
    print("Params:", params)

    print(summary(model, (1, 6, 512)))

    t1 = time.time()
    test_times = 50
    for i in range(test_times):
        output = model(xyz)
    t2 = time.time()
    print("fps:", (1 / ((t2 - t1) / test_times)))