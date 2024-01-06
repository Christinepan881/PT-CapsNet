# This file contains modules common to various models

import math

import numpy as np
import torch
import torch.nn as nn

from utils.datasets import letterbox
from utils.general import non_max_suppression, make_divisible, scale_coords
from models.capslayers import *
#Caps2d

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def DWCaps(ch_in, n_in, ch_out, n_out, k=1, s=1, act=True):
    # Depthwise Capsolution
    return Caps(ch_in, n_in, ch_out, n_out, k, s, act=act)


class Caps(nn.Module):
    '''[b,c,n,h,w]->[b,c',n',h',w']'''
    def __init__(self, ch_in, n_in, ch_out, n_out, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Caps, self).__init__()
        #self.Caps = nn.Caps2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.Caps = Capsules2_2_1_1(ch_in, n_in, ch_out, n_out, 
                    kernel_size=k, stride=s, padding=autopad(k,p),bias=False)
        self.bn = nn.BatchNorm3d(ch_out)
        self.act = nn.Hardswish() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.Caps(x)))

    def fuseforward(self, x):
        return self.act(self.Caps(x))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, ch_in, n_in, ch_out, n_out, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        ch_ = int(ch_out * e)  # hidden channels
        n_ = int(n_out * e)
        self.cv1 = Caps(ch_in, n_in, ch_, n_, 1, 1)
        self.cv2 = Caps(ch_, n_, ch_out, n_out, 3, 1, g=g)
        self.add = shortcut and ch_in==ch_out and n_in==n_out

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, ch_in, n_in, ch_out, n_out, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        ch_ = int(ch_out * e)  # hidden channels
        n_ = int(n_out * e)
        self.cv1 = Caps(ch_in, n_in, ch_, n_, 1, 1)
        #self.cv2 = nn.Caps2d(ch_in, n_in, ch_, n_, 1, 1, bias=False)
        #self.cv3 = nn.Caps2d(ch_, n_, ch_, n_, 1, 1, bias=False)
        self.cv2 = Capsules2_2_1_1(ch_in, n_in, ch_, n_, 1, 1, bias=False)
        self.cv3 = Capsules2_2_1_1(ch_, n_, ch_, n_, 1, 1, bias=False)
        self.cv4 = Caps(2 * ch_, n_, ch_out, n_out, 1, 1)
        self.bn = nn.BatchNorm3d(2 * ch_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(ch_, n_, ch_, n_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        #import pdb
        #pdb.set_trace()
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, ch_in, n_in, ch_out, n_out, k=(5, 9, 13)):
        super(SPP, self).__init__()
        ch_ = ch_out // 2  # hidden channels
        n_ = n_out // 2
        self.cv1 = Caps(ch_in, n_in, ch_, n_, 1, 1)
        self.cv2 = Caps(ch_, n_ * (len(k) + 1), ch_out, n_out, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool3d(kernel_size=(1,x,x), stride=1, padding=(0,x // 2,x // 2)) for x in k])
        #self.m = nn.ModuleList([SelfAttentionRouting(ch_, n_,kernel_size=(1,x,x), stride=1, padding=(0,x // 2,x // 2)) for x in k])

    def forward(self, x):
        x = self.cv1(x)

        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 2))


class Focus(nn.Module):
    # [ch_in=1,n_in=3] ->(slice)-> [ch_mid=4,n_mid=3] ->(Caps)-> [ch_out=16,n_out=4]
    def __init__(self, ch_in, n_in, ch_out, n_out, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.Caps = Caps(ch_in*4, n_in,  ch_out, n_out, k, s, p, g, act)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2) #x[b,3,640,640] -> 4*[b,3,320,320] -> [b,4,3,320,320] -> [b,c_,n_,320,320]
        return self.Caps(torch.stack([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class NMS(nn.Module):
    # Non-Maximum Suppression (NMS) module
    conf = 0.25  # confidence threshold
    iou = 0.45  # IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self):
        super(NMS, self).__init__()

    def forward(self, x):
        return non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)


class autoShape(nn.Module):
    # input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    img_size = 640  # inference size (pixels)
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self, model):
        super(autoShape, self).__init__()
        self.model = model

    def forward(self, x, size=640, augment=False, profile=False):
        # supports inference from various sources. For height=720, width=1280, RGB images example inputs are:
        #   opencv:     x = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(720,1280,3)
        #   PIL:        x = Image.open('image.jpg')  # HWC x(720,1280,3)
        #   numpy:      x = np.zeros((720,1280,3))  # HWC
        #   torch:      x = torch.zeros(16,3,720,1280)  # BCHW
        #   multiple:   x = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        p = next(self.model.parameters())  # for device and type
        if isinstance(x, torch.Tensor):  # torch
            return self.model(x.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        if not isinstance(x, list):
            x = [x]
        shape0, shape1 = [], []  # image and inference shapes
        batch = range(len(x))  # batch size
        for i in batch:
            x[i] = np.array(x[i])  # to numpy
            x[i] = x[i][:, :, :3] if x[i].ndim == 3 else np.tile(x[i][:, :, None], 3)  # enforce 3ch input
            s = x[i].shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(x[i], new_shape=shape1, auto=False)[0] for i in batch]  # pad
        x = np.stack(x, 0) if batch[-1] else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255.  # uint8 to fp16/32

        # Inference
        x = self.model(x, augment, profile)  # forward
        x = non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)  # NMS

        # Post-process
        for i in batch:
            if x[i] is not None:
                x[i][:, :4] = scale_coords(shape1, x[i][:, :4], shape0[i])
        return x


class Flatten(nn.Module):
    # Use after nn.AdaptiveAvgPool2d(1) to remove last 2 dimensions
    @staticmethod
    def forward(x):
        return x.view(x.size(0), -1)


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, ch_in, n_in, ch_out, n_out, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool3d((n_in,1,1))  # to x(b,c1,n1,1,1)
        self.Caps = Capsules2_2_1_1(ch_in, n_in, ch_out, 1, 
                        kernel_size=k, stride=s, padding=autopad(k,p), bias=False)# to x(b,c2,1,1,1)
        #self.Caps = nn.Caps2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)  # to x(b,c2,1,1)
        self.flat = Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.Caps(z))  # flatten to x(b,c2)
