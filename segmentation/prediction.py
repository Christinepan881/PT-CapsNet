
import torch
from torchvision import transforms
from PIL import Image

import argparse
import os
import numpy as np
from tqdm import tqdm
from deeplab import *
from datasets import *

def decode_segmap(label_mask, dataset):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    if dataset == 'airbem' or dataset == 'airbem_old_aug' or dataset == 'airbemold':
        n_classes = 4
        label_colours = get_airbem_labels()
    elif dataset == '210506' or '210804':
        n_classes = 3
        label_colours = get_airbemnew_labels()
    elif dataset == 'pascal' or dataset == 'coco':
        n_classes = 21
        label_colours = get_pascal_labels()  
    elif dataset == 'cityscapes':
        n_classes = 19
        label_colours = get_cityscapes_labels()
    else:
        raise NotImplementedError

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r 
    rgb[:, :, 1] = g 
    rgb[:, :, 2] = b 
    rgb = rgb.astype('uint8')
    return rgb #np.array

def get_airbem_labels():
    return np.asarray([[0, 0, 0], [128,0,0], [0,128,0], [128,128,0]])
def get_airbemnew_labels():
    return np.asarray([[0, 0, 0], [128,0,0], [0,128,0]])

#5-AirBEM-semantic/pytorch-deeplab-xception/run/airbemold/deeplab-resnet/model_best.pth.tar
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--train_set', type=str, default='train_0,train_1,train_2,train_3')
    parser.add_argument('--val_set', type=str, default='train_4')
    parser.add_argument('--source', type=str, default='ir')
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='airbemold',
                        #choices=['pascal', 'coco', 'cityscapes', 'airbem', 'airbemold', 'airbem_old_aug'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--use-sbd', action='store_true', default=True,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=513,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=False,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=1,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=1,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    parser.add_argument('--data_dir', type=str, default="/data1/chenbin/AirBEM/20210303/ir",
                        help='the directory of the IR data')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='the directory of the IR data')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    args = parser.parse_args()

    model = DeepLab(num_classes=3,
                        backbone=args.backbone,
                        output_stride=16,
                        sync_bn=False,
                        freeze_bn=False)
    
    
    checkpoint = torch.load(args.resume)
    
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda()
    model.eval()

    data_dir = args.data_dir
    save_dir = args.save_dir
    os.makedirs(save_dir,exist_ok=True)
    imgs = os.listdir(data_dir)
    num = len(imgs)
    for i in tqdm(range(num)):
        cur_ir = Image.open(os.path.join(data_dir,imgs[i]))#.resize((args.crop_size,args.crop_size)) 
        transform = transforms.Compose([
                transforms.Resize(size=(args.crop_size,args.crop_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        cur_ir = transform(cur_ir).unsqueeze(0).cuda() #[1,c,h,w]
        
        with torch.no_grad():
            output = model(cur_ir)

        output = output.squeeze(0) #[4,h,w]
        pred = output.data.cpu().numpy() #[4,h,w]
        pred = np.argmax(pred, axis=0) #[h,w]
        mask = decode_segmap(pred, '210804') #np[h,w,3]
        mask = Image.fromarray(mask)

        img_save = os.path.join(save_dir,imgs[i].split('.')[0]+'.png')
        mask.save(img_save)
    