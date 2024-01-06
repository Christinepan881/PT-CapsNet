import torch
import torch.nn as nn
import torch.nn.functional as F
#from ..initializer import create_initializer
from .ISARCaps import *

class BasicBlock(nn.Module):
    def __init__(self, ch_in,n_in,ch_out,n_out, stride, drop_rate):
        super().__init__()

        self.drop_rate = drop_rate

        self._preactivate_both = (ch_in != ch_out) or (n_in != n_out)

        self.bn1 = nn.BatchNorm3d(ch_in)
        self.conv1 = Capsules2_2_1_1(ch_in,n_in,ch_out,n_in,
                                    kernel_size=3,
                                    stride=stride,  # downsample with first conv
                                    padding=1,
                                    bias=False)

        self.bn2 = nn.BatchNorm3d(ch_out)
        self.conv2 = Capsules2_2_1_1(ch_out,n_in,ch_out,n_out,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    bias=False)

        self.shortcut = nn.Sequential()
        if self._preactivate_both:
            self.shortcut.add_module(
                'caps_shortcut',
                Capsules2_2_1_1(ch_in,n_in,ch_out,n_out, 
                                kernel_size=1,
                                stride=stride,  # downsample
                                padding=0,
                                bias=False))

    def forward(self, x):
        if self._preactivate_both:
            x = F.relu(self.bn1(x),
                       inplace=True)  # shortcut after preactivation
            y = self.conv1(x)
        else:
            y = F.relu(self.bn1(x),
                       inplace=True)  # preactivation only for residual path
            y = self.conv1(y)
        if self.drop_rate > 0:
            y = F.dropout(y,
                          p=self.drop_rate,
                          training=self.training,
                          inplace=False)

        y = F.relu(self.bn2(y), inplace=True)
        y = self.conv2(y)
        y += self.shortcut(x)
        return y


class Network(nn.Module):
    def __init__(self, config):
        super().__init__()

        model_config = config.model.caps2211_wrn
        depth = model_config.depth #28
        #initial_channels = model_config.initial_channels #16
        widening_factor = model_config.widening_factor #10
        drop_rate = model_config.drop_rate #0

        block = BasicBlock
        n_blocks_per_stage = (depth - 4) // 6 #4
        assert n_blocks_per_stage * 6 + 4 == depth

        self.conv = Capsules2_2_1_1(config.dataset.n_channels,1, 16,2, #n_channels[0][0], n_channels[0][1],
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    bias=False)

        self.stage1 = self._make_stage(16,2, #n_channels[0][0], n_channels[0][1], 
                                       80,4, #n_channels[1][0], n_channels[1][1],
                                       n_blocks_per_stage,
                                       block,
                                       stride=1,
                                       drop_rate=drop_rate)
        self.stage2 = self._make_stage(80,4, #n_channels[1][0], n_channels[1][1], 
                                       80,8, #n_channels[2][0], n_channels[2][1],
                                       n_blocks_per_stage,
                                       block,
                                       stride=2,
                                       drop_rate=drop_rate)
        self.stage3 = self._make_stage(80,8, #n_channels[2][0], n_channels[2][1], 
                                       160,8, #n_channels[3][0], n_channels[3][1],
                                       n_blocks_per_stage,
                                       block,
                                       stride=2,
                                       drop_rate=drop_rate)
        self.bn = nn.BatchNorm3d(160) #n_channels[3][0])
        self.SAR = SelfAttentionRouting(16,8,8,1)
        self.fc = Capsules2_2_1_1(160,8, config.dataset.n_classes,1, kernel_size=1,bias=False)

    def _make_stage(self, ch_in,n_in,ch_out,n_out, n_blocks, block, stride,
                    drop_rate):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_name = f'block{index + 1}'
            if index == 0:
                stage.add_module(
                    block_name,
                    block(ch_in,n_in,
                          ch_out,n_out,
                          stride=stride,
                          drop_rate=drop_rate))
            else:
                stage.add_module(
                    block_name,
                    block(ch_out,n_out,
                          ch_out,n_out,
                          stride=1,
                          drop_rate=drop_rate))
        return stage

    def forward(self, x):
        x = x.unsqueeze(2)
        x = self.conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.relu(self.bn(x), inplace=True)

        x = self.SAR(x)
        x = self.fc(x)
        #x = self.shrink(x)
        return x.squeeze(-1).squeeze(-1).squeeze(-1)
