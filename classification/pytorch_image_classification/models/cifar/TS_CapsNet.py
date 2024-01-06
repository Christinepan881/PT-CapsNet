import torch
from torch import nn
from torch.nn import functional as F

from .utils import (
    squash,
    Swish,
    MemoryEfficientSwish,
)


class CapsuleLayer(nn.Module):
    ''' [b,c,n,h,w]->[b,c',n',h',w']''' 
    def __init__(self, ch_in, n_in, ch_out, n_out, kernel_size=3, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.ch_in = ch_in
        self.n_in = n_in
        self.ch_out = ch_out
        self.n_out = n_out 
        
        self.conv_vector = nn.Conv2d(ch_in*n_in, ch_in*n_out, kernel_size=1, groups=ch_in, bias=bias) 
        self.conv_channel = nn.Conv2d(n_out*ch_in, n_out*ch_out, kernel_size=kernel_size, groups=n_out,
                        stride=stride, padding=padding, dilation=dilation, bias=bias) 

        #self._init_weight()
    def forward(self, x):
        h,w = x.shape[-2:] #[b,c,n,h,w]
        n_vote = self.conv_vector( x.reshape(-1,self.ch_in*self.n_in,h,w) )#[b,cn',h,w]
        n_vote = n_vote.reshape(-1,self.ch_in,self.n_out,h,w).permute(0,2,1,3,4).reshape(-1, self.n_out*self.ch_in, h, w) #[b,n'*c,h,w]
    
        c_map = self.conv_channel(n_vote) #[b,n'*c',h',w']
        h1,w1 = c_map.shape[-2:]
        c_map = c_map.reshape(-1, self.n_out, self.ch_out, h1, w1).permute(0,2,1,3,4)
  
        return c_map #[b,c',n',h',w']


    #def _init_weight(self):
    #    for m in self.modules():
    #        if isinstance(m, nn.Conv2d):
    #            #torch.nn.init.kaiming_normal_(m.weight)
    #            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


class CapsuleFCLayer(nn.Module):
    '''[b,c,n]->[b,c',n']'''
    def __init__(self, ch_in, n_in, ch_out, n_out):
        super().__init__()
        self.ch_in = ch_in
        self.n_in = n_in
        self.ch_out = ch_out
        self.n_out = n_out 

        self.pose_trans = nn.Conv1d(ch_in*n_in, ch_in*n_out, 1, groups=ch_in)
        self.caps_trans = nn.Conv1d(n_out*ch_in, n_out*ch_out, 1, groups=n_out)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.pose_trans(x.reshape(B,-1,1)) #[b,cn',1]
        x = self.caps_trans(x.reshape(B,self.ch_in,self.n_out).permute(0,2,1).reshape(B,-1,1)) #[b,n'c',1]
        x = x.reshape(B,self.n_out,self.ch_out).permute(0,2,1) #[b,c',n']
        return x


###########################
###      CapsUnits      ###
###########################
class StartCaps1(nn.Module):
    '''[b,c,n,h,w]->[b,c',n',h',w']'''
    def __init__(self, ch_in, n_in, ch_out, n_out, kernel_size=3, stride=1, padding=0, dilation=1, bias=False, act=MemoryEfficientSwish):
        super().__init__()
        self.caps = CapsuleLayer(ch_in, n_in, ch_out, n_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm3d(ch_out)
        self.act = act()
    
    def forward(self, x):
        x = self.act(self.bn(self.caps(x)))
        return x 


class StartCaps2(nn.Module):
    '''[b,c,n,h,w]->[b,c',n',h',w']'''
    def __init__(self, ch_in, n_in, ch_out, n_out, kernel_size=3, stride=1, padding=0, dilation=1, bias=False, act=MemoryEfficientSwish, ch_n_exp=1):
        super().__init__()
        self.caps1 = CapsuleLayer(ch_in, n_in, ch_out*ch_n_exp, n_out*ch_n_exp, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(ch_out*ch_n_exp)
        self.act1 = act()

        self.caps2 = CapsuleLayer(ch_out*ch_n_exp, n_out*ch_n_exp, ch_out, n_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.bn2 = nn.BatchNorm3d(ch_out)
        self.act2 = act()
    
    def forward(self, x):
        x = self.act1(self.bn1(self.caps1(x)))
        x = self.act2(self.bn2(self.caps2(x)))
        return x 


class SkipCaps(nn.Module):
    '''[b,c,n,h,w]->[b,c',n',h',w']'''
    def __init__(self, ch_in, n_in, ch_out, n_out, out_size, bias=False, act=MemoryEfficientSwish, pool_size=1, ch_n_exp=1):
        super().__init__()
        self.oh, self.ow = (out_size, out_size) if isinstance(out_size, int) else out_size
        self.pool = nn.AdaptiveAvgPool2d(pool_size)

        self.caps1 = CapsuleLayer(ch_in, n_in, ch_out*ch_n_exp, n_out*ch_n_exp, kernel_size=1, bias=False)
        self.act = act()

        self.caps2 = CapsuleLayer(ch_out*ch_n_exp, n_out*ch_n_exp, ch_out, n_out, kernel_size=1, bias=False)
    
    def forward(self, x):
        B,C,N,H,W = x.shape
        x = self.pool(x.reshape(B,C*N,H,W)) #[b,cn,1,1]
        x = F.upsample(x, size=(self.oh,self.ow), mode="bilinear") #[b,cn,h',w']
        x = self.act(self.caps1(x.reshape(B,C,N,self.oh,self.ow))) #[b,c'',n'',h',w']
        x = F.sigmoid(self.caps2(x)) #[b,c',n',h',w']
        return x


class EndCaps(nn.Module):
    '''[b,c,n,h,w],[b,c,n,h,w]->[b,c',n',h,w]'''
    def __init__(self, ch_in, n_in, ch_out, n_out, act=MemoryEfficientSwish, dropout=False, p=0.5):
        super().__init__()
        self.caps = CapsuleLayer(ch_in, n_in, ch_out, n_out, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm3d(ch_out)
        self.act = act()
        self.dropout = dropout
        self.p = p
    
    def forward(self, x1, x2):
        x = x1*x2
        x = self.act(self.bn(self.caps(x)))
        if self.dropout:
            x = F.dropout(x, p=self.p, inplace=True)
        return x 


########################
###      Modules      ###
########################
class BlockModule1(nn.Module):
    '''[b,c,n,h,w]->[b,c',n',h',w']'''
    def __init__(self,ch_in, n_in, ch_out, n_out, out_size, 
                        kernel_size=3, stride=1, padding=0, dilation=1, bias=False, act=MemoryEfficientSwish, 
                        pool_size=1, ch_n_exp=1,
                        dropout=False, p=0.5):
        super().__init__()
        self.start = StartCaps1(ch_in, n_in, ch_out, n_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, act=act)
        self.skip = SkipCaps(ch_out, n_out, ch_out, n_out, out_size, bias=bias, act=act, pool_size=pool_size, ch_n_exp=ch_n_exp)
        self.end = EndCaps(ch_out, n_out, ch_out, n_out, act=act, dropout=dropout, p=p)
    
    def forward(self, x):
        #[b,c,n,h,w]
        x = self.start(x) #[b,c',n',h',w']
        x_skip = self.skip(x)  #[b,c',n',h',w']
        x = self.end(x, x_skip) #[b,c'',n'',h',w']
        return x


class BlockModule2(nn.Module):
    '''[b,c,n,h,w]->[b,c',n',h',w']'''
    def __init__(self,ch_in, n_in, ch_out, n_out, out_size, 
                        kernel_size=3, stride=1, padding=0, dilation=1, bias=False, act=MemoryEfficientSwish, 
                        pool_size=1, ch_n_exp=1,
                        dropout=False, p=0.5):
        super().__init__()
        self.start = StartCaps2(ch_in, n_in, ch_out, n_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, act=act, ch_n_exp=ch_n_exp)
        self.skip = SkipCaps(ch_out, n_out, ch_out, n_out, out_size, bias=bias, act=act, pool_size=pool_size, ch_n_exp=ch_n_exp)
        self.end = EndCaps(ch_out, n_out, ch_out, n_out, act=act, dropout=dropout, p=p)
    
    def forward(self, x):
        #[b,c,n,h,w]
        x = self.start(x) #[b,c',n',h',w']
        x_skip = self.skip(x)  #[b,c',n',h',w']
        x = self.end(x, x_skip) #[b,c'',n'',h',w']
        return x


class FinalModule(nn.Module):
    '''[b,c,n] -> [b,cls,n] -> [b,cls,1] ->[b,cls]'''
    def __init__(self, ch_in, n_in, num_cls, l2norm=False, act=MemoryEfficientSwish):
        super().__init__()
        self.fc_caps = CapsuleFCLayer(ch_in, n_in, num_cls, n_in)
        self.bn = nn.BatchNorm1d(num_cls)
        self.act = act()


        self.l2norm = l2norm
        self.logit_caps = None
        if not l2norm:
            self.logit_caps = CapsuleFCLayer(num_cls, n_in, num_cls, 1)
    
    def forward(self, x):
        x = self.act(self.bn(self.fc_caps(x))) #[b,cls,n]
        if not self.l2norm:
            x = self.logit_caps(x).squeeze(-1)
            return x
        
        x = torch.sum(torch.square(x), dim=-1, keepdim=False) #[b,cls]
        return x


#######################
###      Model      ###
#######################
class Network(nn.Module):
    def __init__(self, config, num_cls=10, in_channels=3, act=MemoryEfficientSwish, ch_n_exp=1):
        super().__init__()
        #STEM:[b,3,32,32] -> [b,32,32,32]
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(32)
        self.act = act()

        #BLOCK1:[b,32,1,32,32] -> [b,32,4,32,32]
        self.block1 = BlockModule1(32, 1, 32, 4, 32, 
                        kernel_size=3, stride=1, padding=1, bias=False, act=act, 
                        pool_size=1, ch_n_exp=ch_n_exp,
                        dropout=False)
        
        #BLOCK2:[b,32,4,32,32] -> [b,32,8,16,16]
        self.block2_1 = BlockModule2(32,4,32,8, 16, 
                        kernel_size=3, stride=2, padding=1, bias=False, act=act, 
                        pool_size=1, ch_n_exp=ch_n_exp,
                        dropout=False)
        self.block2_2 = BlockModule2(32,8,32,8, 16, 
                        kernel_size=3, stride=1, padding=1, bias=False, act=act,
                        pool_size=1, ch_n_exp=ch_n_exp,
                        dropout=True)   

        #BLOCK3:[b,32,8,16,16] -> [b,64,8,16,16]
        self.block3_1 = BlockModule2(32,8,64,8, 16, 
                        kernel_size=3, stride=1, padding=1, bias=False, act=act, 
                        pool_size=1, ch_n_exp=ch_n_exp,
                        dropout=False)
        self.block3_2 = BlockModule2(64,8,64,8, 16, 
                        kernel_size=3, stride=1, padding=1, bias=False, act=act,
                        pool_size=1, ch_n_exp=ch_n_exp,
                        dropout=True) 

        #BLOCK4:[b,64,8,16,16] -> [b,128,8,8,8]
        self.block4_1 = BlockModule2(64,8,128,8, 8, 
                        kernel_size=3, stride=2, padding=1, bias=False, act=act, 
                        pool_size=1, ch_n_exp=ch_n_exp,
                        dropout=False)
        self.block4_2 = BlockModule2(128,8,128,8, 8, 
                        kernel_size=3, stride=1, padding=1, bias=False, act=act,
                        pool_size=1, ch_n_exp=ch_n_exp,
                        dropout=True)
        self.block4_3 = BlockModule2(128,8,128,8, 8, 
                        kernel_size=3, stride=1, padding=1, bias=False, act=act,
                        pool_size=1, ch_n_exp=ch_n_exp,
                        dropout=True)
        
        #[b,128,16,4,4] -> [b,128,16,1,1] -> [b,128,16] -> [b,cls,16] -> [b,cls,1] ->[b,cls]
        #self.final = FinalModule(128, 16, num_cls, l2norm=False, act=act)
        self.final = FinalModule(128, 8, num_cls, l2norm=False, act=act)

    def forward(self, x):
        #[b,3,32,32]
        x = self.act(self.bn(self.stem(x))).unsqueeze(2) #[b,32,1,32,32]

        #BLOCK1:[b,32,1,32,32] -> [b,32,4,32,32]
        x = self.block1(x)

        #x1 = x
        x = self.block2_1(x)
        x = self.block2_2(x)
        #x = x + x1

        #x2 = x
        x = self.block3_1(x)
        x = self.block3_2(x)
        #x = x + x2

        #x3 = x
        x = self.block4_1(x)
        x = self.block4_2(x)
        #x = x + x3
        #x4_0 = x 
        x = self.block4_3(x)
        #x = x + x4_0

        #x = F.adaptive_avg_pool3d(x,(16,1,1)).squeeze(-1).squeeze(-1) #[b,128,16]
        x = F.adaptive_avg_pool3d(x,(8,1,1)).squeeze(-1).squeeze(-1) #[b,128,8]
        x = self.final(x) #[b,cls]
        return x

