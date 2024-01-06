
import torch
import torch.nn as nn
import torch.nn.functional as F
#from ..initializer import create_initializer
from .ISARCaps import *

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, ch_in,n_in,ch_out,n_out, stride):
        super().__init__()
        # downsample with first caps
        self.caps1 = Capsules2_2_1_1(ch_in,n_in,ch_out,n_in,
                    kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(ch_out)
        self.caps2 = Capsules2_2_1_1(ch_out,n_in,ch_out,n_out,
                    kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(ch_out)

        self.shortcut = nn.Sequential()
        if ch_in != ch_out or n_in != n_out:
            self.shortcut.add_module(
                'downsample_caps1x1',
                Capsules2_2_1_1(ch_in,n_in,ch_out,n_out,
                    kernel_size=1, stride=stride, padding=0, bias=False)
            )
            
            self.shortcut.add_module('downsample_bn', nn.BatchNorm3d(ch_out))  # BN

    def forward(self, x):
        y = F.relu(self.bn1(self.caps1(x)), inplace=True)
        y = self.bn2(self.caps2(y))
        y += self.shortcut(x)
        y = F.relu(y, inplace=True)  # apply ReLU after addition
        return y


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, ch_in,n_in,ch_out,n_out, stride):
        super().__init__()

        bottleneck_channels = ch_out // self.expansion
        self.caps1 = Capsules2_2_1_1(ch_in,n_in,bottleneck_channels,n_in,
                    kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)

        # downsample with 3x3 caps
        self.caps2 = Capsules2_2_1_1(bottleneck_channels,n_in,bottleneck_channels,n_out,
                    kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(bottleneck_channels)

        self.caps3 = Capsules2_2_1_1(bottleneck_channels,n_out,ch_out,n_out,
                    kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm3d(ch_out)

        self.shortcut = nn.Sequential()  # identity
        if ch_in != ch_out or n_in != n_out:
            self.shortcut.add_module(
                'downsample_caps1x1',
                Capsules2_2_1_1(ch_in,n_in,ch_out,n_out,
                    kernel_size=1, stride=stride, padding=0, bias=False)
                )
            self.shortcut.add_module('bn', nn.BatchNorm3d(ch_out))  # BN

    def forward(self, x):
        y = F.relu(self.bn1(self.caps1(x)), inplace=True)
        y = F.relu(self.bn2(self.caps2(y)), inplace=True)
        y = self.bn3(self.caps3(y))  # not apply ReLU
        y += self.shortcut(x)
        y = F.relu(y, inplace=True)  # apply ReLU after addition
        return y


class Network(nn.Module):
    def __init__(self, config):
        super().__init__()

        model_config = config.model.caps2211_resnet
        depth = model_config.depth
        initial_ch = model_config.initial_channels
        block_type = model_config.block_type

        assert block_type in ['basic', 'bottleneck']
        if block_type == 'basic':
            block = BasicBlock
            n_blocks_per_stage = (depth - 2) // 6
            assert n_blocks_per_stage * 6 + 2 == depth
        else:
            block = BottleneckBlock
            n_blocks_per_stage = (depth - 2) // 9
            assert n_blocks_per_stage * 9 + 2 == depth

        chs = [
            initial_ch,
            initial_ch * 2 * block.expansion,
            initial_ch * 2 * block.expansion,
        ]
        ns = [4,4,8]

        self.caps = Capsules2_2_1_1(config.dataset.n_channels,1,
                              chs[0], ns[0],
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias=False)
        self.bn = nn.BatchNorm3d(chs[0])

        self.stage1 = self._make_stage(chs[0],ns[0],
                                       chs[0],ns[0],
                                       n_blocks_per_stage,
                                       block,
                                       stride=1)
        self.stage2 = self._make_stage(chs[0],ns[0],
                                       chs[1],ns[1],
                                       n_blocks_per_stage,
                                       block,
                                       stride=2)
        self.stage3 = self._make_stage(chs[1],ns[1],
                                       chs[2],ns[2],
                                       n_blocks_per_stage,
                                       block,
                                       stride=2)
        
        self.SAR = SelfAttentionRouting(16,8,8,1)#[b,16,8,1,1]
        self.fc = Capsules2_2_1_1(16, 8, config.dataset.n_classes, 1,
                    kernel_size=1, bias=False)

    def _make_stage(self, ch_in,n_in,ch_out,n_out, n_blocks, block, stride):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_name = f'block{index + 1}'
            if index == 0:
                stage.add_module(
                    block_name, block(ch_in,n_in,ch_out,n_out,
                                      stride=stride))
            else:
                stage.add_module(block_name,
                                 block(ch_out,n_out,ch_out,n_out, stride=1))
        return stage

    def forward(self, x):
        x = x.unsqueeze(2)
        x = F.relu(self.bn(self.caps(x)), inplace=True)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        x = self.SAR(x)
        x = self.fc(x)

        return x.squeeze(-1).squeeze(-1).squeeze(-1)

