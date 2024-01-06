from __future__ import print_function, division
import os
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import random

import torch
from torchvision import transforms
from torch.utils.data import Dataset


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img,
                'label': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return {'image': img,
                'label': mask}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'label': mask}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        rotate_degree = random.uniform(-1*self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {'image': img,
                'label': mask}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'image': img,
                'label': mask}


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}

class FixResize(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        img = img.resize((self.crop_size, self.crop_size), Image.BILINEAR)
        mask = mask.resize((self.crop_size, self.crop_size), Image.NEAREST)

        return {'image': img,
                'label': mask}

class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}


class AirBEMSegmentation(Dataset):
    """
    AirBEM dataset
    """
    NUM_CLASSES = 4
    def __init__(self, args, base_dir='/data1/chenbin/AirBEM/dataset_semantic', split='train_val_noaug'):

        self.base_dir = base_dir
        self.args = args
        self.split = None
        self.mode = None
        if "train_val" in split:
            self.split = "train_val_noaug"
        else:
            self.split = "test_noaug"
        
        if "test" in split:
            self.mode = "test"
        else:
            self.mode = "train"

        
        buildings = ['doors','walls','windows']
        self.img_sets = []
        for building in buildings:
            img_sets_txt = open(os.path.join(self.base_dir, building, 'ImageSets/Segmentation', self.split+'.txt'))
            cur_sets = img_sets_txt.readlines()
            cur_sets = [(building, line.split('\n')[0]) for line in cur_sets] #[('doors', 'FLIR0002_augmentaion0')..]
            self.img_sets = self.img_sets + cur_sets
            img_sets_txt.close()
        random.shuffle(self.img_sets)
    
    def __len__(self):
        return len(self.img_sets)
    
    def __getitem__(self, idx):
        building, img_id = self.img_sets[idx] #'doors', 'FLIR0002_augmentaion0'
        img = Image.open( os.path.join(self.base_dir, building, 'JPEGImages', img_id+'.jpg') ) #PIL[h,w,3]
        mask = Image.open( os.path.join(self.base_dir, building, 'SegmentationClass', img_id+'.png') ) #PIL[h,w,3]

        return self._preprocess(building, img, mask)
    
    def _preprocess(self, building, img, mask):
        mask = np.array(mask)
        mask = mask/128
        mask = mask[:,:,0] + 10 * mask[:,:,1] #[h,w]
        if building=='windows':
            mask = np.where(mask==1,1,mask)
            mask = np.where(mask==11,2,mask)
            mask = np.where(mask==10,3,mask)
        else:
            mask = np.where(mask==1,1,mask)
            mask = np.where(mask==10,2,mask)
            mask = np.where(mask==11,3,mask)
        mask = Image.fromarray(mask)

        sample = {'image':img, 'label':mask}
        transform = None
        if self.mode == "test": 
            transform = transforms.Compose([
                FixResize(crop_size=self.args.crop_size),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensor()
            ])
        else: 
            transform = transforms.Compose([
                RandomHorizontalFlip(),
                FixResize(crop_size=self.args.crop_size),
                #RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
                RandomGaussianBlur(),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensor()
            ])
        
        return transform(sample)

    def __str__(self):
        return 'AirBEM(split=' + str(self.split) + ')'


class AirBEMVideoSegmentation(Dataset):
    """
    AirBEM dataset
    """
    NUM_CLASSES = 4
    def __init__(self, args, base_dir='/data1/chenbin/AirBEM/videos-frames'):

        self.base_dir = base_dir
        self.args = args

        self.img_sets = os.listdir(base_dir)
        random.shuffle(self.img_sets)
    
    def __len__(self):
        return len(self.img_sets)
    
    def __getitem__(self, idx):
        img_id = self.img_sets[idx] #'FLIR0002_xxx.jpg'
        img = Image.open( os.path.join(self.base_dir, img_id) ) #PIL[h,w,3]
        img = img.resize((self.args.crop_size,self.args.crop_size), Image.BILINEAR)
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ])
        
        return transform(img)

    def __str__(self):
        return 'AirBEMVideo(split=test)'



class AirBEMNew(Dataset):
    """
    AirBEM dataset:2021.03.03
    all:    1372
    usable: 824 -> 1166
    train:  576 -> 815
    test:   248 -> 351
    """
    NUM_CLASSES = 3
    def __init__(self, args, base_dir='/data1/chenbin/AirBEM/20210303', mode='train'):

        self.base_dir = base_dir
        self.args = args
        self.mode = mode
        
        #'/data1/chenbin/AirBEM/20210303/Masks/train'
        self.img_sets = os.listdir(os.path.join(base_dir, 'Masks', mode)) #['FLIR0025.png'...]
        random.shuffle(self.img_sets)
    
    def __len__(self):
        return len(self.img_sets)
    
    def __getitem__(self, idx):
        img_id = self.img_sets[idx] #'FLIR0002.png'
        img = Image.open( os.path.join(self.base_dir, 'IR_Images', img_id.split('.')[0]+'.jpg') ) #PIL[h,w,3]
        mask = Image.open( os.path.join(self.base_dir, 'Masks', self.mode, img_id) ) #PIL[h,w,3]

        return self._preprocess(img, mask)
    
    def _preprocess(self, img, mask): #Red[128,0,0], Green[0,128,0]
        mask = np.array(mask)
        mask = mask/128
        mask = mask[:,:,0] + 10 * mask[:,:,1] #0, r:1, g:10

        #mask = np.where(mask==1,1,mask)
        mask = np.where(mask==10,2,mask)

        mask = Image.fromarray(mask) #[h,w]

        sample = {'image':img, 'label':mask}
        transform = None
        if self.mode == "test": 
            transform = transforms.Compose([
                FixResize(crop_size=self.args.crop_size),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensor()
            ])
        else: 
            transform = transforms.Compose([
                RandomHorizontalFlip(),
                FixResize(crop_size=self.args.crop_size),
                #RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
                RandomGaussianBlur(),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensor()
            ])
        
        return transform(sample)

    def __str__(self):
        return 'AirBEM(mode=' + str(self.mode) + ')'


class AirBEMTesting(Dataset):
    """
    AirBEM dataset:2021.03.03
    testing1
    """
    NUM_CLASSES = 3
    def __init__(self, args, base_dir='/data1/chenbin/AirBEM/20210303/testing1'):

        self.base_dir = base_dir
        self.args = args
        
        #'/data1/chenbin/AirBEM/20210303/Masks/train'
        self.img_sets = os.listdir(base_dir) #['FLIR0025.jpg'...]
        random.shuffle(self.img_sets)
    
    def __len__(self):
        return len(self.img_sets)
    
    def __getitem__(self, idx):
        img_id = self.img_sets[idx] #'FLIR0002.jpg'
        img = Image.open( os.path.join(self.base_dir, img_id) ) #PIL[h,w,3]
        mask = Image.open( os.path.join(self.base_dir, img_id) ) #PIL[h,w,3]

        return self._preprocess(img, mask)
    
    def _preprocess(self, img, mask): #Red[128,0,0], Green[0,128,0]
        mask = np.array(mask)
        mask = mask[:,:,0] 
        mask = Image.fromarray(mask) #[h,w]

        sample = {'image':img, 'label':mask}
        transform = None

        transform = transforms.Compose([
            FixResize(crop_size=self.args.crop_size),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()
        ])

        
        return transform(sample)

    def __str__(self):
        return 'AirBEM(mode=testing1)'

class AirBEM210506(Dataset):
    """
    AirBEM dataset:2021.05.06
    all:   
    good:  1922
    train: 1345
    test:  577
    """
    NUM_CLASSES = 3
    def __init__(self, args, base_dir='/data1/chenbin/AirBEM/20210303', mode='train', form="ir"):

        self.base_dir = base_dir
        self.args = args
        self.mode = mode #"train","test","predict-train","predict-test"
        self.form = form #ir12rgb3 ir csv
        
        #'/data1/chenbin/AirBEM/20210303/Masks/train'
        if "predict" in mode:
            self.img_sets = os.listdir(os.path.join(base_dir, 'mask', mode[8:]))
        else:
            self.img_sets = os.listdir(os.path.join(base_dir, 'mask', mode)) #['FLIR0025.png'...]
        random.shuffle(self.img_sets)
    
    def __len__(self):
        return len(self.img_sets)
    
    def __getitem__(self, idx):
        img_id = self.img_sets[idx] #'FLIR0002.png'
        img = None
        if self.form == "ir":
            img = Image.open( os.path.join(self.base_dir, "ir", img_id.split('.')[0]+'.jpg') ) #PIL[h,w,3]
        elif self.form == "ir12rgb3":
            ir = Image.open( os.path.join(self.base_dir, "ir", img_id.split('.')[0]+'.jpg') ) #PIL[h,w,3]
            rgb = Image.open( os.path.join(self.base_dir, "rgb", img_id.split('.')[0]+'.jpg') ) #PIL[h,w,3]
            rgb = rgb.resize(ir.size, Image.BILINEAR)
            ir = np.array(ir)
            rgb = np.array(rgb)
            img = ir[:,:,:]
            img[:,:,2] = rgb.mean(2)
            img = Image.fromarray(img)
        else:
            raise NotImplementedError

        if "predict" in self.mode:
            mask = Image.open( os.path.join(self.base_dir, 'mask', self.mode[8:], img_id) ) #PIL[h,w,3]
        else:
            mask = Image.open( os.path.join(self.base_dir, 'mask', self.mode, img_id) ) #PIL[h,w,3]
        mask = np.array(mask)
        mask = mask/128
        mask = mask[:,:,0] + 10 * mask[:,:,1] #0, r:1, g:10

        mask = np.where(mask==10,2,mask)

        mask = Image.fromarray(mask) #[h,w]

        return self._preprocess(img, mask)
    
    def _preprocess(self, img, mask): #Red[128,0,0], Green[0,128,0]
        
        sample = {'image':img, 'label':mask}
        transform = None
        if "predict" or "test" in self.mode : 
            transform = transforms.Compose([
                FixResize(crop_size=self.args.crop_size),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensor()
            ])
        else: 
            transform = transforms.Compose([
                RandomHorizontalFlip(),
                #FixResize(crop_size=self.args.crop_size),
                RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
                #RandomGaussianBlur(),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensor()
            ])

        return transform(sample)

    def __str__(self):
        return 'AirBEM(mode=' + str(self.mode) + ', form=' + str(self.form) + ')'

import pickle as pkl
class AirBEM210804(Dataset):
    """
    AirBEM dataset: 2417 (477 test + 1940 train)
    mode: 
        - test: 477
        - train_0:  388
        - train_1:  388
        - train_2:  388
        - train_3:  388
        - train_4:  388
    """
    NUM_CLASSES = 3
    def __init__(self, args, base_dir='/data1/chenbin/AirBEM/20210303', mode='train', form="ir"):

        self.base_dir = base_dir
        self.args = args
        self.mode = mode #"test","train_0","train_1,train_2,train_3,train_4"
        self.form = form #ir12rgb3 ir csv
        
        #'/data1/chenbin/AirBEM/20210303/mask/train'
        #self.img_sets = os.listdir(os.path.join(base_dir, 'mask', mode)) #['FLIR0025.png'...]
        split = os.path.join(base_dir,'split.pkl')
        with open(split, 'rb') as f:
            split = pkl.load(f)
        
        self.ids = []
        if "," in mode:
            modes = mode.split(',')
            for m in modes:
                self.ids = self.ids + split[m]
        else:
            self.ids = split[mode]
       
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        img_id = self.ids[idx] #'FLIR0002'
        img = None
        if self.form == "ir":
            img = Image.open( os.path.join(self.base_dir, "ir", img_id+'.jpg') ) #PIL[h,w,3]
        elif self.form == "ir12rgb3":
            ir = Image.open( os.path.join(self.base_dir, "ir", img_id+'.jpg') ) #PIL[h,w,3]
            rgb = Image.open( os.path.join(self.base_dir, "rgb", img_id+'.jpg') ) #PIL[h,w,3]
            rgb = rgb.resize(ir.size, Image.BILINEAR)
            ir = np.array(ir)
            rgb = np.array(rgb)
            img = ir[:,:,:]
            img[:,:,2] = rgb.mean(2)
            img = Image.fromarray(img)
        else:
            raise NotImplementedError

        mask = Image.open( os.path.join(self.base_dir, 'mask', img_id+'.png') ) #PIL[h,w,3]
        mask = np.array(mask)
        mask = mask/128
        mask = mask[:,:,0] + 10 * mask[:,:,1] #0, r:1, g:10

        mask = np.where(mask==10,2,mask)

        mask = Image.fromarray(mask) #[h,w]

        return self._preprocess(img, mask)
    
    def _preprocess(self, img, mask): #Red[128,0,0], Green[0,128,0]
        
        sample = {'image':img, 'label':mask}
        transform = None
        if "," in self.mode: 
            transform = transforms.Compose([
                RandomHorizontalFlip(),
                #FixResize(crop_size=self.args.crop_size),
                RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
                #RandomGaussianBlur(),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensor()
            ])
        else: 
            transform = transforms.Compose([
                FixResize(crop_size=self.args.crop_size),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensor()
            ])
        
        return transform(sample)

    def __str__(self):
        return 'AirBEM(mode=' + str(self.mode) + ', form=' + str(self.form) + ')'

import glob
class AirBEM220219(Dataset):
    """
    AirBEM dataset: 1400 = 700 ir + 700 rgb
    """
    NUM_CLASSES = 3
    def __init__(self, args, base_dir='/data1/chenbin/AirBEM/DroneCollectedData', mode='test', form="ir"):
        self.base_dir = base_dir
        self.args = args
        self.mode = mode #"test"
        self.form = form #ir rgb ir12rgb3
        #rgb:DJI_0173.jpg ir:DJI_0325_R.JPG
        self.ir_images = glob.glob(os.path.join(base_dir,"*_R.JPG"))
        #'/data1/chenbin/AirBEM/DroneCollectedData/DJI_0325_R.JPG'
       
    def __len__(self):
        return len(self.ir_images)
    
    def __getitem__(self, idx):
        img = Image.open( self.ir_images[idx] ) #PIL[h,w,3]
        img = img.resize((self.args.crop_size,self.args.crop_size), Image.BILINEAR)
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ])
        
        return transform(img)

    def __str__(self):
        return 'AirBEM220219(mode=' + str(self.mode) + ', form=' + str(self.form) + ')'


from torch.utils.data import DataLoader

def make_data_loader(args, **kwargs):
    if args.dataset == 'airbem':
        train_set = AirBEMSegmentation(args, split='train_val_noaug')
        val_set = AirBEMSegmentation(args, split='test_noaug')

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=False,**kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False,**kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == '210804':
        train_set = AirBEM210804(args, mode=args.train_set, form=args.source)
        val_set = AirBEM210804(args, mode=args.val_set, form=args.source)
        test_set = AirBEM210804(args, mode='test', form=args.source)

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=False)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=False)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False)
        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'airbem-video':
        test_set = AirBEMVideoSegmentation(args)

        num_class = test_set.NUM_CLASSES
        train_loader = None
        val_loader = None
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=False,**kwargs)
        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'airbem-new':
        train_set = AirBEMNew(args, mode='train')
        val_set = AirBEMNew(args, mode='test')

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,**kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=True,**kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class
    
    elif args.dataset == 'airbem-testing1':
        train_loader = None
        val_loader = None
        test_set = AirBEMTesting(args)

        num_class = test_set.NUM_CLASSES
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=False,**kwargs)
        
        return train_loader, val_loader, test_loader, num_class
    
    elif args.dataset == '210506':
        train_set = AirBEM210506(args, mode='train', form=args.source)
        val_set = AirBEM210506(args, mode='test', form=args.source)

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,**kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=True,**kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == '220219':
        data_set = AirBEM220219(args, mode='test', form="ir")
        #num_class = train_set.NUM_CLASSES
        test_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, drop_last=False,**kwargs)
        return test_loader#, num_class

    else:
        raise NotImplementedError


