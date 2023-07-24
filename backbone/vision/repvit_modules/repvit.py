import torch.nn as nn

__all__ = ['repvit_m1']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


from timm.models.layers import SqueezeExcite

import torch


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups,
                            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)

    @torch.no_grad()
    def fuse(self):
        if isinstance(self.m, Conv2d_BN):
            m = self.m.fuse()
            assert (m.groups == m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        elif isinstance(self.m, torch.nn.Conv2d):
            m = self.m
            assert (m.groups != m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        else:
            return self


class RepVGGDW(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = Conv2d_BN(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed

    def forward(self, x):
        return self.conv(x) + self.conv1(x) + x

    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1.fuse()

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [1, 1, 1, 1])

        identity = torch.nn.functional.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device),
                                           [1, 1, 1, 1])

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)
        return conv


class RepViTBlock(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(RepViTBlock, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup
        assert (hidden_dim == 2 * inp)

        if stride == 2:
            self.token_mixer = nn.Sequential(
                Conv2d_BN(inp, inp, kernel_size, stride, (kernel_size - 1) // 2, groups=inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
                Conv2d_BN(inp, oup, ks=1, stride=1, pad=0)
            )
            self.channel_mixer = Residual(nn.Sequential(
                # pw
                Conv2d_BN(oup, 2 * oup, 1, 1, 0),
                nn.GELU() if use_hs else nn.GELU(),
                # pw-linear
                Conv2d_BN(2 * oup, oup, 1, 1, 0, bn_weight_init=0),
            ))
        else:
            assert (self.identity)
            self.token_mixer = nn.Sequential(
                RepVGGDW(inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
            )
            self.channel_mixer = Residual(nn.Sequential(
                # pw
                Conv2d_BN(inp, hidden_dim, 1, 1, 0),
                nn.GELU() if use_hs else nn.GELU(),
                # pw-linear
                Conv2d_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0),
            ))

    def forward(self, x):
        return self.channel_mixer(self.token_mixer(x))


from timm.models.vision_transformer import trunc_normal_


class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0), device=l.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Classfier(nn.Module):
    def __init__(self, dim, num_classes, distillation=True):
        super().__init__()
        self.classifier = BN_Linear(dim, num_classes) if num_classes > 0 else torch.nn.Identity()
        self.distillation = distillation
        if distillation:
            self.classifier_dist = BN_Linear(dim, num_classes) if num_classes > 0 else torch.nn.Identity()

    def forward(self, x):
        if self.distillation:
            x = self.classifier(x), self.classifier_dist(x)
            if not self.training:
                x = (x[0] + x[1]) / 2
        else:
            x = self.classifier(x)
        return x

    @torch.no_grad()
    def fuse(self):
        classifier = self.classifier.fuse()
        if self.distillation:
            classifier_dist = self.classifier_dist.fuse()
            classifier.weight += classifier_dist.weight
            classifier.bias += classifier_dist.bias
            classifier.weight /= 2
            classifier.bias /= 2
            return classifier
        else:
            return classifier


class RepViT(nn.Module):
    def __init__(self, cfgs, out_slices, num_classes=1000, distillation=False):
        super(RepViT, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        self.out_slices = out_slices

        # building first layer
        input_channel = self.cfgs[0][2]
        patch_embed = torch.nn.Sequential(Conv2d_BN(3, input_channel // 2, 3, 2, 1), torch.nn.GELU(),
                                          Conv2d_BN(input_channel // 2, input_channel, 3, 2, 1))
        layers = [patch_embed]
        # building inverted residual blocks
        block = RepViTBlock
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.ModuleList(layers)
        # self.classifier = Classfier(output_channel, num_classes, distillation)

    def forward(self, x):
        # x = self.features(x)
        slices_list = []
        for f in self.features:
            x = f(x)
            slices_list.append(x)

        return [slices_list[i] for i in self.out_slices]


from timm.models import register_model


@register_model
def repvit_m1(pretrained=False, num_classes=1000, distillation=False):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s
        [3, 2, 32, 1, 0, 1],
        [3, 2, 32, 0, 0, 1],
        [3, 2, 32, 0, 0, 1],
        [3, 2, 48, 0, 0, 2],
        [3, 2, 48, 1, 0, 1],
        [3, 2, 48, 0, 0, 1],
        [3, 2, 48, 0, 0, 1],
        [3, 2, 96, 0, 1, 2],
        [3, 2, 96, 1, 1, 1],
        [3, 2, 96, 0, 1, 1],
        [3, 2, 96, 1, 1, 1],
        [3, 2, 96, 0, 1, 1],
        [3, 2, 96, 1, 1, 1],
        [3, 2, 96, 0, 1, 1],
        [3, 2, 96, 1, 1, 1],
        [3, 2, 96, 0, 1, 1],
        [3, 2, 96, 1, 1, 1],
        [3, 2, 96, 0, 1, 1],
        [3, 2, 96, 1, 1, 1],
        [3, 2, 96, 0, 1, 1],
        [3, 2, 96, 1, 1, 1],
        [3, 2, 96, 0, 1, 1],
        [3, 2, 96, 0, 1, 1],
        [3, 2, 176, 0, 1, 2],
        [3, 2, 176, 1, 1, 1],
        [3, 2, 176, 0, 1, 1]
    ]
    return RepViT(cfgs, out_slices=[2,6,22,25])


@register_model
def repvit_m2(pretrained=False, num_classes=1000, distillation=False):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s
        [3, 2, 32, 1, 0, 1],
        [3, 2, 32, 0, 0, 1],
        [3, 2, 32, 0, 0, 1],
        [3, 2, 48, 0, 0, 2],
        [3, 2, 48, 1, 0, 1],
        [3, 2, 48, 0, 0, 1],
        [3, 2, 48, 0, 0, 1],
        [3, 2, 120, 0, 1, 2],
        [3, 2, 120, 1, 1, 1],
        [3, 2, 120, 0, 1, 1],
        [3, 2, 120, 1, 1, 1],
        [3, 2, 120, 0, 1, 1],
        [3, 2, 120, 1, 1, 1],
        [3, 2, 120, 0, 1, 1],
        [3, 2, 120, 1, 1, 1],
        [3, 2, 120, 0, 1, 1],
        [3, 2, 120, 1, 1, 1],
        [3, 2, 120, 0, 1, 1],
        [3, 2, 120, 1, 1, 1],
        [3, 2, 120, 0, 1, 1],
        [3, 2, 120, 0, 1, 1],
        [3, 2, 224, 0, 1, 2],
        [3, 2, 224, 1, 1, 1],
        [3, 2, 224, 0, 1, 1]
    ]
    return RepViT(cfgs, out_slices=[2,6,20,24])


@register_model
def repvit_m3(pretrained=False, num_classes=1000, distillation=False):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s
        [3, 2, 32, 1, 0, 1],
        [3, 2, 32, 0, 0, 1],
        [3, 2, 32, 1, 0, 1],
        [3, 2, 32, 0, 0, 1],
        [3, 2, 32, 0, 0, 1],
        [3, 2, 64, 0, 0, 2],
        [3, 2, 64, 1, 0, 1],
        [3, 2, 64, 0, 0, 1],
        [3, 2, 64, 1, 0, 1],
        [3, 2, 64, 0, 0, 1],
        [3, 2, 64, 0, 0, 1],
        [3, 2, 144, 0, 1, 2],
        [3, 2, 144, 1, 1, 1],
        [3, 2, 144, 0, 1, 1],
        [3, 2, 144, 1, 1, 1],
        [3, 2, 144, 0, 1, 1],
        [3, 2, 144, 1, 1, 1],
        [3, 2, 144, 0, 1, 1],
        [3, 2, 144, 1, 1, 1],
        [3, 2, 144, 0, 1, 1],
        [3, 2, 144, 1, 1, 1],
        [3, 2, 144, 0, 1, 1],
        [3, 2, 144, 1, 1, 1],
        [3, 2, 144, 0, 1, 1],
        [3, 2, 144, 1, 1, 1],
        [3, 2, 144, 0, 1, 1],
        [3, 2, 144, 1, 1, 1],
        [3, 2, 144, 0, 1, 1],
        [3, 2, 144, 1, 1, 1],
        [3, 2, 144, 0, 1, 1],
        [3, 2, 144, 0, 1, 1],
        [3, 2, 288, 0, 1, 2],
        [3, 2, 288, 1, 1, 1],
        [3, 2, 288, 0, 1, 1]
    ]
    return RepViT(cfgs, out_slices=[4, 10, 30, 34])


if __name__ == '__main__':
    input_map = torch.randn(1, 3, 320, 320)
    model = repvit_m3()
    output_map = model(input_map)
    for map in output_map:
        print(map.shape)