import torch
import math
import torch.nn as nn
import torch.nn.functional as F

def squash(x, axis): #[b,c,d,w,w]
    xq = torch.sum(torch.square(x), dim=axis, keepdim=True)
    return (xq / (1 + xq)) * (x / torch.sqrt(xq + 1e-7))

def squash_scalar(x, axis): #[b,c,d,w,w]
    xq = torch.sum(torch.square(x), dim=axis, keepdim=True)
    return torch.sqrt(xq) / (1 + xq)

class Capsules2_2_1_1(nn.Module):
    ''' [b,c,n,h,w]->[b,c',n',h,w] dim转换用repeat''' 
    #[b,c,n,h,w]->[b,c,n',h,w]->[b,c',n',h,w] dim转换用repeat 
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

class SelfAttentionRouting_slow(nn.Module):
    def __init__(self,ch_in,n_in,w_in,w_out ):
        super().__init__()
        self.conv1 = nn.Conv2d(ch_in*n_in,ch_in,kernel_size=1,groups=ch_in)
        self.conv2 = nn.Conv1d(ch_in*w_in*w_in,ch_in,kernel_size=1,groups=ch_in)
        self.ch_in = ch_in
        self.n_in = n_in
        self.w_in = w_in
        self.w_out = w_out
    
    def forward(self, x):
        #[b,c,n,w,w]->[b,c,n,w',w']
        caps_mask = self.conv1(x.reshape(-1, self.ch_in*self.n_in, self.w_in, self.w_in)) #[b,c,w,w]
        caps_mask = F.softmax(caps_mask.reshape(-1,self.ch_in,self.w_in*self.w_in), dim=1)#[b,c,ww]
        caps_mask = caps_mask.reshape(-1,self.ch_in,self.w_in,self.w_in).unsqueeze(2).repeat(1,1,self.n_in,1,1)#[b,c,n,w,w]
        x = caps_mask * x
        #[b,cww,n]
        tmp = x.reshape(-1,self.ch_in,self.n_in,self.w_in*self.w_in).permute(0,1,3,2).reshape(-1,self.ch_in*self.w_in*self.w_in,self.n_in)
        vec_mask = F.softmax(self.conv2(tmp), dim=1) #[b,c,n]
        vec_mask = vec_mask.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,self.w_in,self.w_in)
        x = vec_mask * x 

        x = F.adaptive_avg_pool3d(x, output_size=[self.n_in,self.w_out,self.w_out]) 
        return x


class SelfAttentionRouting_cplct(nn.Module):
    def __init__(self,ch_in,n_in,w_in,w_out ):
        super().__init__()
        self.mask = nn.Conv2d(ch_in*n_in,ch_in,kernel_size=1,groups=ch_in)
        
        self.ch_in = ch_in
        self.n_in = n_in
        self.w_in = w_in
        self.w_out = w_out
    
    def forward(self, x):
        #[b,c,n,w,w]->[b,c,n,1,1]
        mask = self.mask(x.reshape(-1,self.ch_in*self.n_in,self.w_in,self.w_in))#[b,c,w,w]
        mask_x = mask.unsqueeze(2).repeat(1,1,self.n_in,1,1) * x #[b,c,n,w,w]
        mask_x = F.avg_pool3d(mask_x, (1,self.w_in,self.w_in))#[b,c,n,1,1]
        return squash(mask_x, 2)#[b,c,n,1,1]
        


class SelfAttentionRouting(nn.Module):
    def __init__(self,ch_in,n_in,w_in,w_out ):
        super().__init__()
        self.mask = nn.Conv2d(ch_in*n_in,ch_in,kernel_size=1,groups=ch_in)
        
        self.ch_in = ch_in
        self.n_in = n_in
        self.w_in = w_in
        self.w_out = w_out
    
    def forward(self, x):
        #[b,c,n,w,w]->[b,c,n,1,1]
        mask = self.mask(x.reshape(-1,self.ch_in*self.n_in,self.w_in,self.w_in))#[b,c,w,w]
        mask_x = mask.reshape(-1,self.ch_in,self.w_in,self.w_in).unsqueeze(2) * x #[b,c,n,w,w]
        
        mask_x = F.adaptive_avg_pool3d(mask_x, (self.n_in,self.w_out,self.w_out)) #[b,c,n,w',w']

        if self.w_out == 1:
            return squash(mask_x,2)
        
        else:
            mask2 = F.adaptive_avg_pool3d(mask_x,(self.n_in,1,1))#[b,c,n,1,1]
            mask2 = squash_scalar(mask2, 2)#[b,c,1,1,1]
            mask2_x = mask2 * mask_x #[b,c,n,w',w']
            return mask2_x
