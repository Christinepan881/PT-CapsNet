
import torch
import torch.nn as nn
class Capsules1_1v_1_1(nn.Module):
    ''' [b,c,n,h,w]->[b,c',n',h,w] dim转换用repeat''' 
    def __init__(self, ch_in, n_in, ch_out, n_out, kernel_size=3, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.ch_in = ch_in
        self.n_in = n_in
        self.ch_out = ch_out
        self.n_out = n_out 
        self.conv_channel = nn.Conv2d(n_in*ch_in, n_in*ch_out, kernel_size=kernel_size, groups=n_in,
                        stride=stride, padding=padding, dilation=dilation, bias=bias) 
        self.conv_vector = nn.Conv2d(n_in, n_out, kernel_size=1, bias=bias) 
        #self._init_weight()
    def forward(self, x):
        h,w = x.shape[-2:] #[2, 256, 8, 33, 33] 
        x = x.permute(0,2,1,3,4).reshape(-1, self.n_in*self.ch_in, h, w) #[b,n*c,h,w]
    
        c_map = self.conv_channel(x) #[b,n*c',h',w']
        h1,w1 = c_map.shape[-2:]
        c_map = c_map.reshape(-1, self.n_in, self.ch_out, h1, w1) #[b,n,c',h',w'][2, 8, 32, 33, 33]

        n_vote = [self.conv_vector( c_map[:,:,i] ).unsqueeze(1) for i in range(self.ch_out)] #c'*[b,1,n',h',w']
        
        return torch.cat(n_vote, dim=1) #[b,c',n',h',w']

    #def _init_weight(self):
    #    for m in self.modules():
    #        if isinstance(m, nn.Conv2d):
    #            #torch.nn.init.kaiming_normal_(m.weight)
    #            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

import torch.nn.functional as F
from ..initializer import create_initializer


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, ch_in,n_in,ch_out,n_out, stride):
        super().__init__()
        # downsample with first caps
        self.caps1 = Capsules1_1v_1_1(ch_in,n_in,ch_out,n_in,
                    kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(ch_out)
        self.caps2 = Capsules1_1v_1_1(ch_out,n_in,ch_out,n_out,
                    kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(ch_out)

        self.shortcut = nn.Sequential()
        if ch_in != ch_out or n_in != n_out:
            self.shortcut.add_module(
                'downsample_caps1x1',
                Capsules1_1v_1_1(ch_in,n_in,ch_out,n_out,
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
        self.caps1 = Capsules1_1v_1_1(ch_in,n_in,bottleneck_channels,n_in,
                    kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)

        # downsample with 3x3 caps
        self.caps2 = Capsules1_1v_1_1(bottleneck_channels,n_in,bottleneck_channels,n_out,
                    kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(bottleneck_channels)

        self.caps3 = Capsules1_1v_1_1(bottleneck_channels,n_out,ch_out,n_out,
                    kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm3d(ch_out)

        self.shortcut = nn.Sequential()  # identity
        if ch_in != ch_out or n_in != n_out:
            self.shortcut.add_module(
                'downsample_caps1x1',
                Capsules1_1v_1_1(ch_in,n_in,ch_out,n_out,
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

        model_config = config.model.caps_resnet
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

        self.caps = Capsules1_1v_1_1(config.dataset.n_channels,1,
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

        # compute conv feature size
        with torch.no_grad():
            dummy_data = torch.zeros(
                (1, config.dataset.n_channels, config.dataset.image_size,
                 config.dataset.image_size),
                dtype=torch.float32)
            self.fe_h, self.fe_w = self._forward_caps(dummy_data).shape[-2:]
            #[b,c,n,h,w]
            #view(-1).shape[0]
        self.shrink = Capsules1_1v_1_1(chs[2]* block.expansion,8,chs[2]* block.expansion,8,
                        kernel_size=(self.fe_h, self.fe_w), bias=False) #[b,c,n,1,1]
        self.fc1 = Capsules1_1v_1_1(chs[2]* block.expansion,8,config.dataset.n_classes,8,
                    kernel_size=1, bias=False)
        self.fc2 = Capsules1_1v_1_1(config.dataset.n_classes,8,config.dataset.n_classes,1,
                    kernel_size=1, bias=False)#[b,num,1,1,1]

        # initialize weights
        initializer = create_initializer(config.model.init_mode)
        self.apply(initializer)

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

    def _forward_caps(self, x):
        x = x.unsqueeze(2)
        x = F.relu(self.bn(self.caps(x)), inplace=True)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        #x = F.adaptive_avg_pool2d(x, output_size=1)
        return x

    def forward(self, x):
        x = self._forward_caps(x)
        x = self.shrink(x)#[b,c,n,1,1]
        x = self.fc1(x)#[b,num,n,1,1]
        x = self.fc2(x)#[b,num,1,1,1]
        return x.squeeze(-1).squeeze(-1).squeeze(-1)
