import numpy as np
import torch
from torch import flatten, nn
from torch.nn import init
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn import functional as F
import time
from torchinfo import summary
from thop import profile
from thop import clever_format


class ContextAttention(nn.Module):

    def __init__(self, dim=512, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size

        self.key_embed = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=4, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.value_embed=nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim)
        )

        factor = 4
        self.attention_embed = nn.Sequential(
            nn.Conv2d(2*dim, 2*dim//factor, 1, bias=False),
            nn.BatchNorm2d(2*dim//factor),
            nn.ReLU(),
            nn.Conv2d(2*dim//factor, kernel_size*kernel_size*dim, 1)
        )


    def forward(self, x):
        bs, c, h, w = x.shape
        k1 = self.key_embed(x)  # bs,c,h,w
        v = self.value_embed(x).view(bs, c, -1)  # bs,c,h,w

        y = torch.cat([k1, x], dim=1)  # bs,2c,h,w
        att = self.attention_embed(y)  # bs, c*k*k,h,w
        att = att.reshape(bs, c, self.kernel_size*self.kernel_size, h, w)
        att = att.mean(2, keepdim=False).view(bs, c, -1)  # bs,c,h*w
        k2 = F.softmax(att, dim=-1)*v
        k2 = k2.view(bs, c, h, w)
        return k1+k2


if __name__ == '__main__':
    input = torch.randn(1, 672, 10, 10).cuda()
    cot = ContextAttention(dim=672, kernel_size=3).cuda()
    output = cot(input)
    print(output.shape)

    print(summary(cot, input_size=[(1, 672, 10, 10)]))
    macs, params = profile(cot, inputs=[input])
    macs *= 2
    macs, params = clever_format([macs, params], "%.3f")
    print("FLOPs:", macs)
    print("Params:", params)

    t1 = time.time()
    test_times = 300
    for i in range(test_times):
        output = cot(input)
    t2 = time.time()
    print("fps:", (1 / ((t2 - t1) / test_times)))
