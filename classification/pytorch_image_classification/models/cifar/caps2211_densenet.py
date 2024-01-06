import torch
import torch.nn as nn
import torch.nn.functional as F
from ..initializer import create_initializer
from .ISARCaps import *

class BasicBlock(nn.Module):
    '''ch_in -> ch_in+ch_out, no size change'''
    def __init__(self, ch_in, n_in, ch_out, n_out, drop_rate):
        super().__init__()

        self.drop_rate = drop_rate

        self.bn = nn.BatchNorm3d(ch_in)
        self.conv = Capsules2_2_1_1(ch_in,n_in,
                              ch_out,n_out,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias=False)

    def forward(self, x):
        y = self.conv(F.relu(self.bn(x), inplace=True))
        if self.drop_rate > 0:
            y = F.dropout(y,
                          p=self.drop_rate,
                          training=self.training,
                          inplace=False)
        return torch.cat([x, y], dim=1)


class BottleneckBlock(nn.Module):
    '''ch_in -> ch_in+ch_out, no size change'''
    def __init__(self, ch_in, n_in, ch_out, n_out, drop_rate):
        super().__init__()

        self.drop_rate = drop_rate

        bottleneck_channels = ch_out * 4

        self.bn1 = nn.BatchNorm3d(ch_in)
        self.conv1 = Capsules2_2_1_1(ch_in,n_in,
                               bottleneck_channels,n_in,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)

        self.bn2 = nn.BatchNorm3d(bottleneck_channels)
        self.conv2 = Capsules2_2_1_1(bottleneck_channels,n_in,
                               ch_out,n_out,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)

    def forward(self, x):
        y = self.conv1(F.relu(self.bn1(x), inplace=True))
        if self.drop_rate > 0:
            y = F.dropout(y,
                          p=self.drop_rate,
                          training=self.training,
                          inplace=False)
        y = self.conv2(F.relu(self.bn2(y), inplace=True))
        if self.drop_rate > 0:
            y = F.dropout(y,
                          p=self.drop_rate,
                          training=self.training,
                          inplace=False)
        return torch.cat([x, y], dim=1)


class TransitionBlock(nn.Module):
    '''ch_in -> ch_out, size/2'''
    def __init__(self, ch_in, n_in, ch_out, n_out, drop_rate):
        super().__init__()

        self.drop_rate = drop_rate

        self.bn = nn.BatchNorm3d(ch_in)
        self.conv = Capsules2_2_1_1(ch_in,n_in,
                              ch_out,n_out,
                              kernel_size=1,
                              stride=1,
                              padding=0,
                              bias=False)
        
        self.shrink = Capsules2_2_1_1(ch_out,n_out,
                                      ch_out,n_out,
                                      kernel_size=2,
                                      stride=2,
                                      bias=False)

    def forward(self, x):
        x = self.conv(F.relu(self.bn(x), inplace=True))
        if self.drop_rate > 0:
            x = F.dropout(x,
                          p=self.drop_rate,
                          training=self.training,
                          inplace=False)
        x = self.shrink(x)
        return x


class Network(nn.Module):
    def __init__(self, config):
        super().__init__()

        model_config = config.model.caps2211_densenet
        depth = model_config.depth #100
        block_type = model_config.block_type #bottleneck
        self.growth_rate = model_config.growth_rate #12
        self.drop_rate = model_config.drop_rate #0
        self.compression_rate = model_config.compression_rate #0.5

        assert block_type in ['basic', 'bottleneck']
        if block_type == 'basic':
            block = BasicBlock
            n_blocks_per_stage = (depth - 4) // 3
            assert n_blocks_per_stage * 3 + 4 == depth
        else:
            block = BottleneckBlock
            n_blocks_per_stage = (depth - 4) // 6 #16
            assert n_blocks_per_stage * 6 + 4 == depth

        self.n_in = 3 #4 #3
        ch_in = [2 * self.growth_rate] #16,72,100,228 #24,18,15,27
        for index in range(3):
            denseblock_ch_out = int(ch_in[-1] +
                                          n_blocks_per_stage *
                                          self.growth_rate) #144 200 228 #36 30 27
            if index < 2:
                transitionblock_ch_out = int(denseblock_ch_out *
                                                   self.compression_rate) #72 100 #18 15
            else:
                transitionblock_ch_out = denseblock_ch_out #228 #27
            ch_in.append(transitionblock_ch_out)

        self.conv = Capsules2_2_1_1(config.dataset.n_channels,1,
                              ch_in[0],self.n_in,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias=False)
        self.stage1 = self._make_stage(ch_in[0],self.n_in, n_blocks_per_stage,
                                       block, True)
        self.stage2 = self._make_stage(ch_in[1],self.n_in, n_blocks_per_stage,
                                       block, True)
        self.stage3 = self._make_stage(ch_in[2],self.n_in, n_blocks_per_stage,
                                       block, False)
        self.bn = nn.BatchNorm3d(ch_in[3])
        self.shrink = SelfAttentionRouting(16,8,8,1)

        self.fc = Capsules2_2_1_1(ch_in[3],self.n_in, config.dataset.n_classes,1, kernel_size=1,bias=False)

        # initialize weights
        initializer = create_initializer(config.model.init_mode)
        self.apply(initializer)

        

    def _make_stage(self, ch_in,n_in, n_blocks, block, add_transition_block):
        stage = nn.Sequential()
        for index in range(n_blocks):
            stage.add_module(
                f'block{index + 1}',
                block(ch_in + index * self.growth_rate, n_in, self.growth_rate, n_in, 
                      self.drop_rate))
        if add_transition_block:
            ch_in = int(ch_in + n_blocks * self.growth_rate)
            ch_out = int(ch_in * self.compression_rate)
            stage.add_module(
                'transition',
                TransitionBlock(ch_in, n_in, ch_out, n_in, self.drop_rate))
        return stage

    def forward(self, x):
        x = self.conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)#[b,228,8,8,8]
        x = F.relu(self.bn(x), inplace=True)
        #x = F.adaptive_avg_pool3d(x, output_size=(self.n_in,1,1))#[b,228,8,1,1]
        x = self.shrink(x)
        x = self.fc(x)#[b,num,1,1,1]
        return x.squeeze(-1).squeeze(-1).squeeze(-1)
