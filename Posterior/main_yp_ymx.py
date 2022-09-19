import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import transforms as transforms
import torchvision
from torch.utils.data import Dataset, DataLoader

from datetime import datetime
from utils.utils import create_if_not_exists as cine
from utils.utils import Tee
from utils.utils import vis_square
import pickle
from fcn import *
from fcn_shallow import *
from heatmap import *
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from math import cos, pi
import random

parser = argparse.ArgumentParser(description='Posterior Training')
parser.add_argument('--data_pascub_images', metavar='DIR',
                    help='path to pascub images')
parser.add_argument('--data_pascub_parts', metavar='DIR',
                    help='path to pascub parts')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--model_path', default='', type=str, metavar='PATH',
                    help='path to the pretrained model')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--lr_decay', default='cos', type=str,
                    help='lr decay type')
parser.add_argument('--crop_size', default='256', type=int,
                    help='size of cropped image')
parser.add_argument('--visualize', dest='visualize', action='store_true',
                    help='visualize middle output')
parser.add_argument('--nparts', default='15', type=int,
                    help='number of keypoints')

best_loss = 10000
time_string = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

cine('logs')
Tee('logs/cmd_log_{}'.format(time_string), 'w')

unisize = 256
outsize = 256

resnet_transform = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

class MultiEpochsDataLoader(torch.utils.data.DataLoader):

     def __init__(self, *args, **kwargs):
         super().__init__(*args, **kwargs)
         self._DataLoader__initialized = False
         self.batch_sampler = _RepeatSampler(self.batch_sampler)
         self._DataLoader__initialized = True
         self.iterator = super().__iter__()

     def __len__(self):
         return len(self.batch_sampler.sampler)

     def __iter__(self):
         for i in range(len(self)):
             yield next(self.iterator)


class _RepeatSampler(object):
     """ Sampler that repeats forever.

     Args:
         sampler (Sampler)
     """

     def __init__(self, sampler):
         self.sampler = sampler

     def __iter__(self):
         while True:
             yield from iter(self.sampler)

def load_train_annos():
    train_annos = torchfile.load('./anno/train.dat')
    annos = {}
    for name, kp in train_annos.items():
        name = name.decode()
        annos[name] = {}
        for idx, loc in kp.items():
            annos[name][int(idx.decode())] = tuple(loc)
    return annos

def load_val_annos():
    val_annos = torchfile.load('./anno/val.dat')
    annos = {}
    for name, kp in val_annos.items():
        name = name.decode()
        annos[name] = {}
        for idx, loc in kp.items():
            annos[name][int(idx.decode())] = tuple(loc)
    return annos

def gaussian(img, pt):
    sigma = 8
    # Draw a 2D gaussian

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] > img.shape[1] or ul[1] >= img.shape[0] or
        br[0] < 0 or br[1] < 0):
    # If not, just return the image as is
        return img

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img

list_ex = ['Ivory_Gull_0085_49456.jpg', 'Ivory_Gull_0040_49180.jpg', 'Mallard_0130_76836.jpg', 'White_Necked_Raven_0070_102645.jpg', 'Clark_Nutcracker_0020_85099.jpg', 'Pelagic_Cormorant_0022_23802.jpg', 'Western_Gull_0002_54825.jpg', 'Brewer_Blackbird_0028_2682.jpg']

def make_gaussian(paths, pt):
        (h,w,_) = pt 
        anno_train = load_train_annos()
        anno_val = load_val_annos()
        annos = {**anno_train, **anno_val}
        anno = annos[os.path.basename(paths)[:-3]+'jpg']
        masks = np.zeros((15, h, w), dtype=np.float32)
        for idx in range(15):
            if int(anno[idx+1][2]) == 1:
                masks[idx] = gaussian(masks[idx], (int(round(anno[idx+1][0]*w)), int(round(anno[idx+1][1]*h))))

        return masks

class ImageLabelDataset(Dataset):
    def __init__(
            self,
            img_path_list,
            label_path_list,
            img_size=(128, 128),
            mode=True,
    ):
        self.img_path_list = img_path_list
        self.label_path_list = label_path_list
        self.img_size = img_size
        self.mode = mode

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        im_path = self.img_path_list[index]
        lbl_path = self.label_path_list[index]
        im = np.array(Image.open(im_path))
        try:
            lbl = np.load(lbl_path)
        except:
            lbl = np.array(Image.open(lbl_path))
        if len(lbl.shape) == 3:
            lbl = lbl[:, :, 0]
        keyps = make_gaussian(im_path, im.shape)

        mask = lbl.copy()
        mask[np.where(mask>0)] = 1

        xmin, xmax, ymin, ymax = np.where(mask!=0)[0].min(), np.where(mask!=0)[0].max(), np.where(mask!=0)[1].min(), np.where(mask!=0)[1].max()
        
        shift1 = 20
        shift2 = 20
        shift3 = 20
        shift4 = 20

        img_new = im[max(0,xmin-shift1):min(xmax+shift2,im.shape[0]),max(0,ymin-shift3):min(ymax+shift4, im.shape[1])]
        mask_new = mask[max(0,xmin-shift1):min(xmax+shift2,im.shape[0]),max(0,ymin-shift3):min(ymax+shift4, im.shape[1])]
        lbl_new = lbl[max(0,xmin-shift1):min(xmax+shift2,im.shape[0]),max(0,ymin-shift3):min(ymax+shift4, im.shape[1])]
        keyps_new = keyps[:,max(0,xmin-shift1):min(xmax+shift2,im.shape[0]),max(0,ymin-shift3):min(ymax+shift4, im.shape[1])]

        im = Image.fromarray(img_new)
        mask = Image.fromarray(mask_new.astype('uint8'))
        lbl = Image.fromarray(lbl_new.astype('uint8'))
        im, lbl, mask, keyps = self.transform(im, lbl, mask, keyps_new.transpose(1,2,0))
        return im, lbl, mask, keyps, im_path

    def transform(self, img, lbl, mask, keyps):
        jitter = random.random() < 0.5
        if jitter and self.mode==True:
            img = torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0)(img)

        hflip = random.random() < 0.5
        if hflip and self.mode==True:
          img = img.transpose(Image.FLIP_LEFT_RIGHT)
          lbl = lbl.transpose(Image.FLIP_LEFT_RIGHT)
          mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
          keyps = cv2.flip(keyps, 1)
          lbl = np.array(lbl)
          lbl_new = lbl.copy()
          lbl_new[np.where(lbl==4)] = 5
          lbl_new[np.where(lbl==6)] = 7
          lbl_new[np.where(lbl==8)] = 9
          lbl_new[np.where(lbl==5)] = 4
          lbl_new[np.where(lbl==7)] = 6
          lbl_new[np.where(lbl==9)] = 8
          lbl = Image.fromarray(lbl_new)
          keyps_new = keyps.copy()
          keyps_new[:,:,6] = keyps[:,:,10]
          keyps_new[:,:,7] = keyps[:,:,11]
          keyps_new[:,:,8] = keyps[:,:,12]
          keyps_new[:,:,10] = keyps[:,:,6]
          keyps_new[:,:,11] = keyps[:,:,7]
          keyps_new[:,:,12] = keyps[:,:,8]
          keyps = keyps_new


        img = img.resize((self.img_size[0], self.img_size[1]))
        lbl = lbl.resize((self.img_size[0], self.img_size[1]), resample=Image.NEAREST)
        mask = mask.resize((self.img_size[0], self.img_size[1]), resample=Image.NEAREST)
        keyps = cv2.resize(keyps, (256,256))
        mask = torch.from_numpy(np.array(mask)).float()
        keyps = torch.from_numpy(np.array(keyps)).float().permute(2,0,1)
        lbl = torch.from_numpy(np.array(lbl)).long()
        img = torchvision.transforms.ToTensor()(img)
        img = resnet_transform(img)
        return img, lbl, mask, keyps

def main():
    global args, best_loss
    args = parser.parse_args()
    print(args)

    global fig, ax1, ax2, ax3, ax4
    if args.visualize:
        plt.ion()
        plt.show()
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)

    model_yp_ykymx = fcn_shallow(input_dim=1, nparts=11)
    model_yp_ykymx = model_yp_ykymx.cuda()
    criterion_yp = nn.CrossEntropyLoss().cuda()
    criterion_l1 = nn.L1Loss().cuda()  

    param_base = [p for (n, p) in model_yp_ykymx.named_parameters() if 'image' not in n]
    
    optimizer_yp_ykymx = torch.optim.SGD([
                                {'params' : param_base},
                                {'params' : model_yp_ykymx.model_image.parameters(), 'lr' : 0.01}],
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
   
    train_sampler = None
    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    images_pascub = []
    labels_pascub = []
    for file in os.listdir(args.data_pascub_images):
        img_path = os.path.join(args.data_pascub_images, file)
        label_path = os.path.join(args.data_pascub_parts, file[:-3]+'png')
        images_pascub.append(img_path)
        labels_pascub.append(label_path)

    train_data = ImageLabelDataset(img_path_list=images_pascub,
                              label_path_list=labels_pascub,
                            img_size=(256, 256), mode=True)

    train_loader_pascub = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    images_pascub = []
    labels_pascub = []
    for file in os.listdir(args.data_pascub_images[:-7] + '_val'):
        img_path = os.path.join(args.data_pascub_images[:-7] + '_val', file)
        label_path = os.path.join(args.data_pascub_parts[:-7] + '_val', file[:-3]+'png')
        images_pascub.append(img_path)
        labels_pascub.append(label_path)

    val_data = ImageLabelDataset(img_path_list=images_pascub,
                              label_path_list=labels_pascub,
                            img_size=(256, 256), mode=False)

    val_loader_pascub = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)


    if args.evaluate:
        model.load_state_dict(torch.load(args.model_path)['state_dict'])
        testdir = os.path.join(args.data, 'test')
        test_dataset = Heatmap(
                testdir, transforms.Compose([
                transforms.ResizeTarget((outsize, outsize)),
                transforms.ToTensor(),
                normalize,
            ]))

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        validate(test_loader, model, criterion, args.epochs-1)
        return

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader_pascub,
             model_yp_ykymx,
             criterion_yp,
             criterion_l1,
             optimizer_yp_ykymx,
             epoch)

        # evaluate on validation set
        loss_val = validate(val_loader_pascub,
             model_yp_ykymx,
             criterion_yp,
             epoch)

        # remember best prec@1 and save checkpoint
        is_best = loss_val < best_loss
        best_loss = min(loss_val, best_loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict_yp_ykymx': model_yp_ykymx.state_dict(),
            'best_loss': best_loss,
            'optimizer_yp_ykymx' : optimizer_yp_ykymx.state_dict(),
        }, is_best)


def train(train_loader_pascub,
             model_yp_ykymx,
             criterion_yp,
             criterion_l1,
             optimizer_yp_ykymx,
             epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_yp = AverageMeter()    
    losses_l1 = AverageMeter()    
    model_yp_ykymx.train()

    end = time.time()
    train_loader_len = len(train_loader_pascub)
    for i, (input_pascub, part, mask, keyps, _) in enumerate(train_loader_pascub):
        adjust_learning_rate(optimizer_yp_ykymx, epoch, i, train_loader_len)
        data_time.update(time.time() - end)

        input_pascub_var = torch.autograd.Variable(input_pascub.cuda())
        part = part.cuda()
        mask = mask.cuda()
        keyps = keyps.cuda()
        part_var = torch.autograd.Variable(part)
        mask_var = torch.autograd.Variable(mask)
        keyps_var = torch.autograd.Variable(keyps)

        yp_out = model_yp_ykymx(torch.cat([input_pascub_var, mask_var.unsqueeze(1)], 1))
        loss_yp = criterion_yp(yp_out, part_var)

        optimizer_yp_ykymx.zero_grad()
        loss_yp.backward()
        optimizer_yp_ykymx.step()

        losses_yp.update(loss_yp.item(), input_pascub.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if args.visualize:
                visualize(inputs.cpu().numpy()[0].transpose(1,2,0),
                        outputs.data.cpu().numpy()[0], 
                        target_var.data.cpu().numpy()[0])


            print('Epoch: [{0}][{1}/{2}] '
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Loss yp {loss_yp.val:.3e} ({loss_yp.avg:.3e}) '.format(
                   epoch, i, len(train_loader_pascub), batch_time=batch_time,
                   data_time=data_time, loss_yp=losses_yp))

       
def validate(val_loader_pascub,
             model_yp_ykymx,
             criterion_yp,
             epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_yp = AverageMeter()

    # switch to evaluate mode
    model_yp_ykymx.eval()
    end = time.time()

    with torch.no_grad():
        for i, (input_pascub, part, mask, keyps, paths) in enumerate(val_loader_pascub):

            input_pascub_var = torch.autograd.Variable(input_pascub.cuda())
            part = part.cuda()
            mask = mask.cuda()
            keyps = keyps.cuda()
            part_var = torch.autograd.Variable(part)
            mask_var = torch.autograd.Variable(mask)
            keyps_var = torch.autograd.Variable(keyps)

            yp_out = model_yp_ykymx(torch.cat([input_pascub_var, mask_var.unsqueeze(1)], 1))

            loss_yp = criterion_yp(yp_out, part_var)
            losses_yp.update(loss_yp.item(), input_pascub.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:

                if args.visualize:
                    visualize(inputs.cpu().numpy()[0].transpose(1,2,0),
                            outputs.data.cpu().numpy()[0], 
                            target_var.data.cpu().numpy()[0])

                print('Test: [{0}/{1}] '
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'Loss yp {loss_yp.val:.3e} ({loss_yp.avg:.3e}) '.format(
                    i, len(val_loader_pascub), batch_time=batch_time,
                    loss_yp=losses_yp))
    return losses_yp.avg

def save_checkpoint(state, is_best, filename='*_checkpoint.pth.tar'):
    work_dir = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
    save_dir = os.path.join('./checkpoints_', work_dir) + "_yp_ymx"
    cine(save_dir)
    epoch = str(state['epoch'])
    filename = filename.replace('*', epoch)
    filename = os.path.join(save_dir, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(save_dir, epoch+'_model_best.pth.tar'))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, iteration, num_iter):
    lr0 = optimizer.param_groups[0]['lr']
    lr1 = optimizer.param_groups[1]['lr']

    warmup_epoch = 0 #if args.warmup else 0
    warmup_iter = warmup_epoch * num_iter
    current_iter = iteration + epoch * num_iter
    max_iter = args.epochs * num_iter

    if args.lr_decay == 'step':
        lr = args.lr * (args.gamma ** ((current_iter - warmup_iter) / (max_iter - warmup_iter)))
    elif args.lr_decay == 'cos':
        lr0 = args.lr * (1 + cos(pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2
        lr1 = lr1 * (1 + cos(pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2
    elif args.lr_decay == 'linear':
        lr = args.lr * (1 - (current_iter - warmup_iter) / (max_iter - warmup_iter))
    elif args.lr_decay == 'schedule':
        count = sum([1 for s in args.schedule if s <= epoch])
        lr = args.lr * pow(args.gamma, count)
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_decay))

    if epoch < warmup_epoch:
        lr = args.lr * current_iter / warmup_iter

    optimizer.param_groups[0]['lr'] = lr0
    optimizer.param_groups[1]['lr'] = lr1


def visualize(img, otpt, grth):
    pred = vis_square(otpt)
    gt = vis_square(grth)
    ax1.imshow(pred)
    ax2.imshow(gt)
    ax3.imshow(pred>0.5)
    img = (img - img.min()) / (img.max() - img.min())
    for j in range(grth.shape[0]):
        enlarge = cv2.resize(pred[j], (unisize, unisize))
        img += np.random.rand(3) * enlarge[...,np.newaxis].repeat(3,axis=2)
    img = (img - img.min()) / (img.max() - img.min())
    ax4.imshow(img)
    plt.draw()
    plt.savefig('test.png')
    plt.pause(0.001)


if __name__ == '__main__':
    main()
