import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ISARCaps import *

class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm):
        super(Decoder, self).__init__()
        self.low_ch_in = None
        if backbone == 'resnet' or backbone == 'drn':
            self.low_ch_in = 32 #256
        elif backbone == 'xception':
            self.low_ch_in = 16 #128
        elif backbone == 'mobilenet':
            self.low_ch_in = 3  #24
        else:
            raise NotImplementedError

        self.caps1 = Capsules2_2_1_1(self.low_ch_in, 8, 18, 8, 1, bias=False)
        self.bn1 = BatchNorm(18)
        self.relu = nn.ReLU()
        self.last_caps = nn.Sequential(Capsules2_2_1_1(50, 8, 32, 8, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(32),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       Capsules2_2_1_1(32, 8, 32, 8, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(32),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       Capsules2_2_1_1(32, 8, num_classes, 1, kernel_size=1, stride=1))
        self._init_weight()


    def forward(self, x, low_level_feat):
        #x[b,32,8,h,w] low[b,c,h',w']
        h, w = low_level_feat.shape[-2:]
        low_level_feat = low_level_feat.reshape(-1, self.low_ch_in, 8, h, w)
        low_level_feat = self.caps1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat) #[b,18,8,h',w']

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='trilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1) #[b,50,8,h',w']
        x = self.last_caps(x).squeeze(2) #[b,cls,1,h',w']->[b,cls,h',w']

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_decoder(num_classes, backbone, BatchNorm):
    return Decoder(num_classes, backbone, BatchNorm)