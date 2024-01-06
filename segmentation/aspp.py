import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ISARCaps import *

class _CapsASPPModule(nn.Module):
    def __init__(self, ch_in, n_in, ch_out, n_out, kernel_size, padding, dilation, BatchNorm):
        super().__init__()
        self.atrous_caps = Capsules2_2_1_1(ch_in, n_in, ch_out, n_out, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(ch_out)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_caps(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, backbone, output_stride, BatchNorm):
        super(ASPP, self).__init__()
        self.ch_in = None
        if backbone == 'drn':
            self.ch_in = 64 #512
        elif backbone == 'mobilenet':
            self.ch_in = 40 #320
        else:
            self.ch_in = 256 #2048
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _CapsASPPModule(self.ch_in, 8, 32, 8, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _CapsASPPModule(self.ch_in, 8, 32, 8, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _CapsASPPModule(self.ch_in, 8, 32, 8, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _CapsASPPModule(self.ch_in, 8, 32, 8, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(SelfAttentionRouting(self.ch_in, 8, 1), #nn.AdaptiveAvgPool3d((8, 1, 1)),
                                             Capsules2_2_1_1(self.ch_in, 8, 32, 8, 1, stride=1, bias=False),
                                             BatchNorm(32),
                                             nn.ReLU())
        self.caps1 = Capsules2_2_1_1(32, 40, 32, 8, 1, bias=False)
        self.bn1 = BatchNorm(32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        #x[b,c,h,w]
        h, w = x.shape[-2:]
        x = x.reshape(-1, self.ch_in, 8, h, w)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='trilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=2)

        x = self.caps1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x) #[b,32,8,h,w]

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_aspp(backbone, output_stride, BatchNorm):
    return ASPP(backbone, output_stride, BatchNorm)