# This file contains experimental modules

import numpy as np
import torch
import torch.nn as nn

from models.common import Caps, DWCaps
from utils.google_utils import attempt_download
from models.capslayers import *


class CrossConv(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, ch_in, n_in, ch_out, n_out, k=3, s=1, g=1, e=1.0, shortcut=False):
        # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
        super(CrossConv, self).__init__()
        ch_ = int(ch_out * e)  # hidden channels
        n_ = int(n_out * e) # hidden channels
        self.cv1 = Conv(ch_in, n_in, ch_, n_, (1, k), (1, s))
        self.cv2 = Conv(ch_, n_, ch_out, n_out, (k, 1), (s, 1), g=g)
        self.add = shortcut and ch_in==ch_out and n_in==n_out

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    # Cross Convolution CSP
    def __init__(self, ch_in, n_in, ch_out, n_out, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        ch_ = int(ch_out * e)  # hidden channels
        n_ = int(n_out * e) # hidden channels
        self.cv1 = Conv(ch_in, n_in, ch_, n_, 1, 1)
        self.cv2 = Capsules2_2_1_1(ch_in, n_in, ch_, n_, 1, 1, bias=False)
        self.cv3 = Capsules2_2_1_1(ch_, n_, ch_, n_, 1, 1, bias=False)
        self.cv4 = Conv(2*ch_, 2*n_, ch_out, n_out, 1, 1)
        self.bn = nn.BatchNorm3d(2 * ch_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[CrossConv(ch_, n_, ch_, n_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class Sum(nn.Module):
    # Weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, n, weight=False):  # n: number of inputs
        super(Sum, self).__init__()
        self.weight = weight  # apply weights boolean
        self.iter = range(n - 1)  # iter object
        if weight:
            self.w = nn.Parameter(-torch.arange(1., n) / 2, requires_grad=True)  # layer weights

    def forward(self, x):
        y = x[0]  # no weight
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + x[i + 1]
        return y


class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, ch_in, n_in, ch_out, n_out, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super(GhostConv, self).__init__()
        ch_ = int(ch_out * e)  # hidden channels
        n_ = int(n_out * e) 
        self.cv1 = Conv(ch_in, n_in, ch_, n_, k, s, None, g, act)
        self.cv2 = Conv(ch_, n_, ch_, n_, 5, 1, None, ch_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)


class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, ch_in, n_in, ch_out, n_out, k, s):
        super(GhostBottleneck, self).__init__()
        ch_ = ch_out // 2
        n_ = n_in // 2
        self.conv = nn.Sequential(GhostConv(ch_in, n_in, ch_, n_, 1, 1),  # pw
                                  DWConv(ch_, n_, ch_, n_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
                                  GhostConv(ch_, ch_out, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
                                      Conv(ch_in, n_in, ch_out, n_out, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class MixConv2d(nn.Module):
    # Mixed Depthwise Conv https://arxiv.org/abs/1907.09595
    def __init__(self, ch_in, n_in, ch_out, n_out, k=(1, 3), s=1, equal_ch=True):
        super(MixConv2d, self).__init__()
        groups = len(k)
        if equal_ch:  # equal ch_ per group
            i = torch.linspace(0, groups - 1E-6, ch_out).floor()  # ch_out indices
            ch_ = [(i == g).sum() for g in range(groups)]  # intermediate channels
        else:  # equal weight.numel() per group
            b = [ch_out] + [0] * groups
            a = np.eye(groups + 1, groups, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            ch_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList([Capsules2_2_1_1(ch_in, n_in, int(ch_[g]), n_in, k[g], s, k[g] // 2, bias=False) for g in range(groups)])
        self.bn = nn.BatchNorm3d(ch_out)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return x + self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))


class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.cat(y, 1)  # nms ensemble
        y = torch.stack(y).mean(0)  # mean ensemble
        return y, None  # inference, train output


def attempt_load(weights, map_location=None):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        attempt_download(w)
        model.append(torch.load(w, map_location=map_location)['model'].float().fuse().eval())  # load FP32 model

    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    if len(model) == 1:
        return model[-1]  # return model
    else:
        print('Ensemble created with %s\n' % weights)
        for k in ['names', 'stride']:
            setattr(model, k, getattr(model[-1], k))
        return model  # return ensemble
