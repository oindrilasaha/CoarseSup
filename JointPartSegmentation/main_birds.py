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
from utils import create_if_not_exists as cine
from utils import Tee
from utils import vis_square
import pickle
from fcn import *
from fcn_posterior import *
from heatmap import *
import matplotlib.pyplot as plt
import numpy as np
from math import cos, pi
import cv2

from PIL import Image
import random

parser = argparse.ArgumentParser(description='JointParts Training')
parser.add_argument('--exp_name', default='kpandmask', type=str,
                    help='name of experiment used to save checkpoints')
parser.add_argument('--keypoint', default=True, type=bool,
                    help='keypoints as coarse supervision')
parser.add_argument('--mask', default=True, type=bool,
                    help='masks as coarse supervision')       
parser.add_argument('--data_cub_images', metavar='DIR',
                    help='path to cub images')
parser.add_argument('--data_cub_parts', metavar='DIR',
                    help='path to cub parts')
parser.add_argument('--data_cub_images_full', metavar='DIR',
                    help='path to cub images')
parser.add_argument('--data_cub_masks_full', metavar='DIR',
                    help='path to cub parts')       
parser.add_argument('--data_pascal_images', metavar='DIR',
                    help='path to pascal images')
parser.add_argument('--data_pascal_parts', metavar='DIR',
                    help='path to pascal parts')        
parser.add_argument('--posterior_ckpt', default=None, type=str,
                    help='path to posterior model checkpoint')     
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=40, type=int, metavar='N',
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
parser.add_argument('--lr_decay', default='cos', type=str,
                    help='lr decay type')
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


palette = [255,  255,  255,
            220, 220, 0,
           190, 153, 153,
            250, 170, 30,
           220, 0, 0,
           107, 142, 35,
           102, 102, 156,
           152, 251, 152,
           119, 11, 32,
           244, 35, 232,
           220, 20, 60,
           52 , 83  ,84,
          194 , 87 , 125,
          225,  96  ,18,
          31 , 102 , 211,
          104 , 131 , 101
          ]

def colorize_mask(mask, palette):
    # mask: numpy array of the mask

    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return np.array(new_mask.convert('RGB'))

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

resnet_transform = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])


class ImageLabelDataset(Dataset):
    def __init__(
            self,
            img_path_list,
            label_path_list,
            mode,
            img_size=(128, 128),
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

        xmin, xmax, ymin, ymax = np.where(lbl!=0)[0].min(), np.where(lbl!=0)[0].max(), np.where(lbl!=0)[1].min(), np.where(lbl!=0)[1].max()
        im_new = im[max(0,xmin-20):min(xmax+20,im.shape[0]),max(0,ymin-20):min(ymax+20, im.shape[1])]
        lbl = lbl[max(0,xmin-20):min(xmax+20,im.shape[0]),max(0,ymin-20):min(ymax+20, im.shape[1])]
        im = Image.fromarray(im_new)

        if len(np.unique(lbl))==1:
            print(im_path)

        lbl = Image.fromarray(lbl.astype('uint8'))
        im, lbl = self.transform(im, lbl)

        return im, lbl, im_path

    def transform(self, img, lbl):
        jitter = random.random() < 0.5
        if jitter and self.mode==True:
            img = torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0)(img)

        hflip = random.random() < 0.5
        if hflip and self.mode==True:
          img = img.transpose(Image.FLIP_LEFT_RIGHT)
          lbl = lbl.transpose(Image.FLIP_LEFT_RIGHT)
          lbl = np.array(lbl)
          lbl_new = lbl.copy()
          lbl_new[np.where(lbl==4)] = 5
          lbl_new[np.where(lbl==6)] = 7
          lbl_new[np.where(lbl==8)] = 9
          lbl_new[np.where(lbl==5)] = 4
          lbl_new[np.where(lbl==7)] = 6
          lbl_new[np.where(lbl==9)] = 8
          lbl = Image.fromarray(lbl_new)


        img = img.resize((self.img_size[0], self.img_size[1]))
        lbl = lbl.resize((self.img_size[0], self.img_size[1]), resample=Image.NEAREST)
        lbl = torch.from_numpy(np.array(lbl)).long()
        img = torchvision.transforms.ToTensor()(img)
        return img, lbl


def main():
    global args, best_loss
    args = parser.parse_args()
    print(args)

    global fig, ax1, ax2, ax3, ax4
    if args.visualize:
        plt.ion()
        plt.show()
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)

    # create model
    model_yp_x = fcn(pretrained=True, nparts=11).cuda()
    model_yk_yp = fcn(input_dim=11, nparts=args.nparts).cuda()
    if args.keypoint==True and args.mask==True:
        input_posterior = 16
        model_name = 'posterior_ykymx'
    elif args.keypoint==False and args.mask==True:
        input_posterior = 1
        model_name = 'posterior_ymx'
    elif args.keypoint==True and args.mask==False:
        input_posterior = 15
        model_name = 'posterior_ykx'

    model_yp_ykymx = fcn_posterior(input_dim=input_posterior, nparts=11).cuda()

    state_dict = torch.load(args.posterior_ckpt)['state_dict_yp_ykymx']
    model_yp_ykymx.load_state_dict(state_dict)

    state_dict = torch.load('../checkpoints_pretrained/finetuned.pth')['model_state_dict']
    state_dict['decoder.4.weight'] = state_dict.pop("decoder.fc.weight")
    state_dict['decoder.4.bias'] = state_dict.pop("decoder.fc.bias")

    model_yp_x.load_state_dict(state_dict)

    model_yk_yp.load_state_dict(torch.load('../checkpoints_pretrained/model_yk_yp.pth.tar')['state_dict'])

    criterion_yk = nn.MSELoss().cuda()   
    criterion_yp = nn.CrossEntropyLoss().cuda()
    criterion_l1 = nn.L1Loss().cuda()  


    optimizer_yp_x = torch.optim.SGD(model_yp_x.parameters(), 0.00001,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    optimizer_yk_yp = torch.optim.SGD(model_yk_yp.parameters(), 0.000000001,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    optimizer_yp_ykymx = torch.optim.SGD(model_yp_ykymx.parameters(), 0.001,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    
    train_sampler = None
    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    train_dataset_cub = Heatmap(
        args.data_cub_images_full,
        args.data_cub_masks_full,
        True,
        transforms.Compose([
            transforms.ResizeTarget((outsize, outsize)),
            transforms.ToTensor(),
            normalize
        ]))


    train_loader_cub = DataLoader(
        train_dataset_cub, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_dataset_cub = Heatmap(
            args.data_cub_images_full,
            args.data_cub_masks_full,
            False,
            transforms.Compose([
            transforms.ResizeTarget((outsize, outsize)),
            transforms.ToTensor(),
            normalize
        ]))

    val_loader_cub = DataLoader(
        val_dataset_cub,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    images_pascal = []
    labels_pascal = []

    img_path_base = args.data_cub_images
    lbl_path_base = args.data_cub_parts
   
    for file in os.listdir(img_path_base):
        img_path = os.path.join(img_path_base, file)
        label_path = os.path.join(lbl_path_base, file[:-3]+'png')
        images_pascal.append(img_path)
        labels_pascal.append(label_path)

    img_path_base = args.data_pascal_images
    lbl_path_base = args.data_pascal_parts

    for file in os.listdir(img_path_base):
        img_path = os.path.join(img_path_base, file)
        label_path = os.path.join(lbl_path_base, file[:-3]+'png')
        images_pascal.append(img_path)
        labels_pascal.append(label_path)

    train_data = ImageLabelDataset(img_path_list=images_pascal,
                              label_path_list=labels_pascal, mode=True,
                            img_size=(256, 256))

    train_loader_pascal = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=args.workers)

    images_pascal = []
    labels_pascal = []

    img_path_base = args.data_cub_images[:-7] + '_val'
    lbl_path_base = args.data_cub_parts[:-7] + '_val'

    for file in os.listdir(img_path_base):
        img_path = os.path.join(img_path_base, file)
        label_path = os.path.join(lbl_path_base, file[:-3]+'png')
        images_pascal.append(img_path)
        labels_pascal.append(label_path)

    val_data = ImageLabelDataset(img_path_list=images_pascal,
                              label_path_list=labels_pascal, mode=False,
                            img_size=(256, 256))

    val_loader_pascal = DataLoader(val_data, batch_size=4, shuffle=False, num_workers=args.workers)


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
        train(train_loader_cub, train_loader_pascal,
             model_yp_x, model_yk_yp, model_yp_ykymx,
             criterion_yk, criterion_yp, criterion_l1,
             optimizer_yp_x, optimizer_yk_yp, optimizer_yp_ykymx,
             epoch)

        # evaluate on validation set
        loss_val = validate(val_loader_cub, val_loader_pascal,
             model_yp_x, model_yk_yp, model_yp_ykymx,
             criterion_yk, criterion_yp, criterion_l1,
             epoch)

        print("Avg Loss = " + str(loss_val))

        # remember best prec@1 and save checkpoint
        is_best = loss_val < best_loss
        best_loss = min(loss_val, best_loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict_yp_x': model_yp_x.state_dict(),
            'state_dict_yk_yp': model_yk_yp.state_dict(),
            'state_dict_yp_ykymx': model_yp_ykymx.state_dict(),
            'best_loss': best_loss,
            'optimizer_yp_x' : optimizer_yp_x.state_dict(),
            'optimizer_yk_yp' : optimizer_yk_yp.state_dict(),
            'optimizer_yp_ykymx' : optimizer_yp_ykymx.state_dict(),
        }, is_best)


def train(train_loader_cub, train_loader_pascal,
             model_yp_x, model_yk_yp, model_yp_ykymx,
             criterion_yk, criterion_yp, criterion_l1,
             optimizer_yp_x, optimizer_yk_yp, optimizer_yp_ykymx,
             epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_yp_x = AverageMeter()
    losses_yp1yp2 = AverageMeter()
    losses_ym = AverageMeter()        
    pck = AverageMeter()
    train_iterator_pascal = iter(train_loader_pascal)

    train_loader_len = len(train_loader_cub)
    end = time.time()
    for i, (input_cub, mask, keyp, paths) in enumerate(train_loader_cub):
        adjust_learning_rate(optimizer_yp_x, epoch, i, train_loader_len)
        adjust_learning_rate(optimizer_yp_ykymx, epoch, i, train_loader_len)
        adjust_learning_rate(optimizer_yk_yp, epoch, i, train_loader_len)
        #### E Phase ###########************************************************************
        model_yp_ykymx.train()
        model_yp_x.eval()
        model_yk_yp.eval()
        # measure data loading time
        data_time.update(time.time() - end)

        keyp = keyp.cuda()
        mask = mask.cuda()
        input_cub_var = torch.autograd.Variable(input_cub.cuda())
        keyp_var = torch.autograd.Variable(keyp)
        mask_var = torch.autograd.Variable(mask)

        if args.keypoint==True and args.mask==True:
            yp1_out = model_yp_ykymx(torch.cat([input_cub_var, keyp_var, mask_var[:,1].unsqueeze(1)], 1))
        elif args.keypoint==False and args.mask==True:
            yp1_out = model_yp_ykymx(torch.cat([input_cub_var, mask_var[:,1].unsqueeze(1)], 1))
        elif args.keypoint==True and args.mask==False:
            yp1_out = model_yp_ykymx(torch.cat([input_cub_var, keyp_var], 1))

        # import pdb;pdb.set_trace()

        yp1_out_softmax = torch.softmax(yp1_out, dim=1)
        
        yk_out = model_yk_yp(yp1_out_softmax)
        ym_out = torch.cat([yp1_out_softmax[:,0].unsqueeze(1), torch.sum(yp1_out_softmax[:,1:], 1).unsqueeze(1)],1)
        
        with torch.no_grad():
            yp2_out = model_yp_x(input_cub_var)

        if args.mask==True:
            loss_ym = criterion_yp(ym_out, mask_var[:,1].long())
        else:
            loss_ym = 0
        if args.keypoint==True:
            loss_yk = criterion_yk(yk_out, keyp_var)
        else:
            loss_yk = 0
        loss_yp1yp2 = criterion_yp(yp1_out, yp2_out.softmax(dim=1))

        entropy = -torch.mean(yp1_out_softmax*torch.log(yp1_out_softmax))

        loss_yp_ykymx = loss_ym + 50*loss_yk + 0.05*loss_yp1yp2 + 0.01*entropy

        optimizer_yp_ykymx.zero_grad()
        loss_yp_ykymx.backward()
        optimizer_yp_ykymx.step()


        #### M Phase ###########************************************************************
        try:
            input_pascal, part, path = next(train_iterator_pascal)
        except:
            train_iterator_pascal = iter(train_loader_pascal)
            input_pascal, part, path = next(train_iterator_pascal)

        input_pascal_var = torch.autograd.Variable(input_pascal.cuda())
        part = part.cuda()
        part_var = torch.autograd.Variable(part)

        model_yp_ykymx.eval()
        model_yp_x.train()
        model_yk_yp.train()

        yp2_out = model_yp_x(input_pascal_var)
        yp2_out_cub = model_yp_x(input_cub_var)
        loss_yp_x = criterion_yp(yp2_out, part_var)

        with torch.no_grad():
            if args.keypoint==True and args.mask==True:
                yp1_out = model_yp_ykymx(torch.cat([input_cub_var, keyp_var, mask_var[:,1].unsqueeze(1)], 1))
            elif args.keypoint==False and args.mask==True:
                yp1_out = model_yp_ykymx(torch.cat([input_cub_var, mask_var[:,1].unsqueeze(1)], 1))
            elif args.keypoint==True and args.mask==False:
                yp1_out = model_yp_ykymx(torch.cat([input_cub_var, keyp_var], 1))

        if args.keypoint==True:
            yk_out = model_yk_yp(torch.softmax(yp1_out, dim=1))
            loss_yk_yp = criterion_yk(yk_out, keyp_var)
        else:
            loss_yk_yp = 0

        loss_yp1yp2 = criterion_yp(yp2_out_cub, yp1_out.softmax(dim=1))


        loss_M = 0.001*loss_yk_yp + 50*loss_yp1yp2 + 0.05*loss_yp_x

        optimizer_yk_yp.zero_grad()
        optimizer_yp_x.zero_grad()
        loss_M.backward()
      
        optimizer_yk_yp.step()
        optimizer_yp_x.step()


        # measure pck and record loss
        if args.keypoint==True:
            pck_score = calc_pck(yk_out.data, keyp)
        else:
            pck_score = 0
        losses_yp_x.update(loss_yp_x.item(), input_cub.size(0))
        losses_yp1yp2.update(loss_yp1yp2.item(), input_cub.size(0))
        losses_ym.update(loss_ym.item(), input_cub.size(0))
        pck.update(pck_score, input_cub.size(0))


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
                  'Loss yp|x {loss_yp_x.val:.3e} ({loss_yp_x.avg:.3e}) '
                  'Loss yp1-yp2 {loss_yp1yp2.val:.3e} ({loss_yp1yp2.avg:.3e}) '
                  'Loss ym|yp {loss_ym.val:.3e} ({loss_ym.avg:.3e}) '
                  'PCK {pck.val:.3f} ({pck.avg:.3f})'.format(
                   epoch, i, len(train_loader_cub), batch_time=batch_time,
                   data_time=data_time, loss_yp_x=losses_yp_x, loss_yp1yp2=losses_yp1yp2, loss_ym=losses_ym , pck=pck))

       
def validate(val_loader_cub, val_loader_pascal,
             model_yp_x, model_yk_yp, model_yp_ykymx,
             criterion_yk, criterion_yp, criterion_l1,
             epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_yp_x = AverageMeter()
    losses_yp1yp2 = AverageMeter()
    losses_ym = AverageMeter()        
    pck = AverageMeter()

    # switch to evaluate mode
    model_yp_ykymx.eval()
    model_yp_x.eval()
    model_yk_yp.eval()

    end = time.time()

    val_iterator_pascal = iter(val_loader_pascal)

    with torch.no_grad():
        for i, (input_cub, mask, keyp, paths) in enumerate(val_loader_cub):
            keyp = keyp.cuda()
            mask = mask.cuda()
            input_cub_var = torch.autograd.Variable(input_cub.cuda())
            keyp_var = torch.autograd.Variable(keyp)
            mask_var = torch.autograd.Variable(mask)

            yp1_out = model_yp_ykymx(torch.cat([input_cub_var, keyp_var], 1))
            yp1_out_softmax = torch.softmax(yp1_out, dim=1)

            yk_out = model_yk_yp(yp1_out_softmax)
            ym_out = torch.sum(yp1_out_softmax[:,1:], 1).unsqueeze(1)

            try:
                input_pascal, part, path = next(val_iterator_pascal)
            except:
                val_iterator_pascal = iter(val_loader_pascal)
                input_pascal, part, path = next(val_iterator_pascal)

            input_pascal_var = torch.autograd.Variable(input_pascal.cuda())
            part = part.cuda()
            part_var = torch.autograd.Variable(part)

            yp2_out = model_yp_x(input_pascal_var)
            yp2_out_cub = model_yp_x(input_cub_var)

            loss_yp_x = criterion_yp(yp2_out, part_var)
            loss_yp1yp2 = criterion_yp(yp1_out, yp2_out_cub)
            loss_ym = criterion_l1(ym_out, mask_var[:,1].unsqueeze(1))
            pck_score = calc_pck(yk_out.data, keyp)

            losses_yp_x.update(loss_yp_x.item(), input_cub.size(0))
            losses_yp1yp2.update(loss_yp1yp2.item(), input_cub.size(0))
            losses_ym.update(loss_ym.item(), input_cub.size(0))
            pck.update(pck_score, input_cub.size(0))

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
                  'Loss yp|x {loss_yp_x.val:.3e} ({loss_yp_x.avg:.3e}) '
                  'Loss yp1-yp2 {loss_yp1yp2.val:.3e} ({loss_yp1yp2.avg:.3e}) '
                  'Loss ym|yp {loss_ym.val:.3e} ({loss_ym.avg:.3e}) '
                  'PCK {pck.val:.3f} ({pck.avg:.3f})'.format(
                    i, len(val_loader_cub), batch_time=batch_time,
                    loss_yp_x=losses_yp_x, loss_yp1yp2=losses_yp1yp2, loss_ym=losses_ym , pck=pck))
                if (i+1)%12==0:
                    return losses_yp_x.avg
    return losses_yp_x.avg

def save_checkpoint(state, is_best, filename='*_checkpoint.pth.tar'):
    work_dir = os.path.basename(os.path.dirname(os.path.realpath(__file__))) + args.exp_name
    save_dir = os.path.join('./checkpoints_', work_dir)
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
    # import pdb;pdb.set_trace()
    lr0 = optimizer.param_groups[0]['lr']

    warmup_epoch = 0 
    warmup_iter = warmup_epoch * num_iter
    current_iter = iteration + epoch * num_iter
    max_iter = args.epochs * num_iter

    if args.lr_decay == 'step':
        lr = args.lr * (args.gamma ** ((current_iter - warmup_iter) / (max_iter - warmup_iter)))
    elif args.lr_decay == 'cos':
        lr0 = lr0 * (1 + cos(pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2
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

def find_max_loc(heatmap):
    (batches, channels) = heatmap.shape[:2]
    locs = np.zeros((batches, channels, 2), np.uint32)
    for b in range(batches):
        for c in range(channels):
            locs[b,c] = np.unravel_index(heatmap[b,c].argmax(), heatmap[b,c].shape)
    
    return locs

def get_dists(preds, gts):
    (batches, channels) = preds.shape[:2]
    dists = np.zeros((channels, batches), np.int32)
    for b in range(batches):
        for c in range(channels):
            if gts[b, c, 0] > 0 and gts[b, c, 1] > 0:
                dists[c,b] = ((gts[b,c] - preds[b,c]) ** 2).sum() ** 0.5
            else:
                dists[c,b] = -1
    return dists

def within_threshold(dist, thr = 0.1):
    dist = dist[dist != -1]
    if len(dist) > 0:
        return (dist < thr * outsize).sum() / float(len(dist))
    else:
        return -1

def calc_pck(output, target):
    output_np = output.cpu().numpy()
    target_np = target.cpu().numpy()
    preds = find_max_loc(output_np)
    gts = find_max_loc(target_np)
    dists = get_dists(preds, gts)
    acc = np.zeros(args.nparts, dtype=np.float32)
    avg_ccc = 0.0
    bad_idx_count = 0

    for i in range(args.nparts):
        acc[i] = within_threshold(dists[i])
        if acc[i] >= 0:
            avg_ccc = avg_ccc + acc[i]
        else:
            bad_idx_count = bad_idx_count + 1
  
    if bad_idx_count == args.nparts:
        return 0
    else:
        return avg_ccc / (args.nparts - bad_idx_count) * 100

def inverse_normalization(img, 
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]):
    
    for i, m, s in zip(img, mean, std):
        i *= s
        i += m
    return img

if __name__ == '__main__':
    main()
