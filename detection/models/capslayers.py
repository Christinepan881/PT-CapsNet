from __future__ import division

import torch 
import torch.nn as nn
import pdb 
#python3 train.py --data voc.yaml --cfg yolov5s.yaml --weights '' --batch-size 64

def squash(x, axis): #[b,c,d,w,w]
    xq = torch.sum(torch.square(x), dim=axis, keepdim=True)
    return (xq / (1 + xq)) * (x / torch.sqrt(xq + 1e-7))

def squash2(x, axis): #[b,c,d,w,w]
    xq = torch.sum(torch.square(x), dim=axis, keepdim=True) #[b,c,1,w,w]
    xq = xq.mean(-1,keepdim=True).mean(-2,keepdim=True) #[b,c,1,1,1]
    return xq / (1 + xq)

class Capsules2_2_1_1(nn.Module):
    #[b,c,n,h,w]->[b,c,n',h,w]->[b,c',n',h,w]
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
        x = self.conv_vector( x.reshape(-1,self.ch_in*self.n_in,h,w) )#[b,cn',h,w]
        x = x.reshape(-1,self.ch_in,self.n_out,h,w).permute(0,2,1,3,4).reshape(-1, self.n_out*self.ch_in, h, w) #[b,n'*c,h,w]
    
        x = self.conv_channel(x) #[b,n'*c',h',w']
        h,w = x.shape[-2:]
  
        return x.reshape(-1, self.n_out, self.ch_out, h, w).permute(0,2,1,3,4) #[b,c',n',h',w']

    #def _init_weight(self):
    #    for m in self.modules():
    #        if isinstance(m, Capsules2_2_1_1):
    #            #torch.nn.init.kaiming_normal_(m.weight)
    #            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

class Capsules2_2_1_1_Linear(nn.Module):
    #[b,c,n]->[b,c,n']->[b,c',n']
    def __init__(self, ch_in, n_in, ch_out, n_out, bias=True):
        super().__init__()
        self.ch_in = ch_in
        self.n_in = n_in
        self.ch_out = ch_out
        self.n_out = n_out 
        
        self.conv_vector = nn.Conv2d(ch_in*n_in, ch_in*n_out, kernel_size=1, groups=ch_in, bias=bias) 
        self.conv_channel = nn.Conv2d(n_out*ch_in, n_out*ch_out, kernel_size=1, groups=n_out, bias=bias) 

        #self._init_weight()
    def forward(self, x):
  
        n_vote = self.conv_vector( x.reshape(-1,self.ch_in*self.n_in).unsqueeze(-1).unsqueeze(-1) )#[b,cn',1,1]
        n_vote = n_vote.reshape(-1,self.ch_in,self.n_out,1,1).permute(0,2,1,3,4).reshape(-1, self.n_out*self.ch_in,1,1) #[b,n'*c,1,1]
    
        c_map = self.conv_channel(n_vote) #[b,n'*c',1,1]
        c_map = c_map.reshape(-1, self.n_out, self.ch_out).permute(0,2,1)
  
        return c_map #[b,c',n']

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, Capsules2_2_1_1):
                #torch.nn.init.kaiming_normal_(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

class SelfAttentionRouting(nn.Module):
    def __init__(self,ch_in,n_in, kernel_size, stride, padding):
        super().__init__()
        self.conv1 = nn.Conv2d(ch_in*n_in,ch_in,kernel_size=1,groups=ch_in)
        self.ch_in = ch_in
        self.n_in = n_in
        self.pool = nn.MaxPool3d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        #[b,c,n,w,w]->[b,c,n,w',w']
        h,w = x.shape[-2:]
        caps_mask = self.conv1(x.reshape(-1, self.ch_in*self.n_in, h,w)) #[b,c,w,w]
        x = caps_mask.unsqueeze(2) * x #[b,c,n,w,w]

        vec_mask = squash2(x,2) #[b,c,1,1,1]
        x = vec_mask * x #[b,c,n,w,w]

        x = self.pool(x)
        return x


if __name__ == "__main__":
    pdb.set_trace()
    pass