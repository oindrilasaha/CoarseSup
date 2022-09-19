import sys
sys.path.append('../')
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
import argparse
import gc
import os
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import glob
import numpy as np
from PIL import Image
import json
from torchvision import transforms
from Posterior.fcn import *
import random

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
        img = transforms.ToTensor()(img)
        return img, lbl

def cross_validate(cp_path, args):
    cp_list = [cp_path]

    data_all = glob.glob(args.data_images + "/*")
    data_all_labels = glob.glob(args.data_parts + "/*")

    images = [path for path in data_all if 'npy' not in path]
    labels = [path for path in data_all_labels if 'npy' not in path]
    images.sort()
    labels.sort()

    ids = range(11)

    fold_num =int( len(images) / 5)
    resnet_transform = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])


    classifier = fcn(pretrained=False, nparts=11)

    classifier.decoder[3] = UpBlock(64, 64, upsample=True)
    classifier.decoder.add_module("fc",nn.Conv2d(64,11,1,1))

    cross_mIOU = []

    for i in range(5):
        val_image = images[fold_num * i: fold_num *i + fold_num]
        val_label = labels[fold_num * i: fold_num *i + fold_num]
        test_image = [img for img in images if img not in val_image]
        test_label =[label for label in labels if label not in val_label]
        print("Val Data length,", str(len(val_image)))
        print("Testing Data length,", str(len(test_image)))

        val_data = ImageLabelDataset(img_path_list=val_image,
                                      label_path_list=val_label, mode=False,
                                      img_size=(256, 256))
        val_data = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0)

        test_data = ImageLabelDataset(img_path_list=test_image,
                                  label_path_list=test_label, mode=False,
                                img_size=(256, 256))
        test_data = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)

        best_miou = 0
        best_val_miou = 0

        criterion = nn.CrossEntropyLoss().cuda()
        # import pdb;pdb.set_trace()
        for resume in cp_list:
            checkpoint = torch.load(resume)
            classifier.load_state_dict(checkpoint['model_state_dict'])
            classifier.cuda()
            classifier.eval()

            unions = {}
            intersections = {}
            for target_num in ids:
                unions[target_num] = 0
                intersections[target_num] = 0
            loss = []
            with torch.no_grad():
                for _, da, in enumerate(val_data):
                    img, mask = da[0], da[1]
                    if img.size(1) == 4:
                        img = img[:, :-1, :, :]
                    img = img.cuda()
                    mask = mask.cuda()
                    input_img_tensor = []
                    for b in range(img.size(0)):
                        input_img_tensor.append(resnet_transform(img[b]))
                    input_img_tensor = torch.stack(input_img_tensor)

                    y_pred = classifier(input_img_tensor)#['out']
                    los = criterion(y_pred, mask)
                    loss.append(los.item())

                    y_pred = torch.log_softmax(y_pred, dim=1)
                    _, y_pred = torch.max(y_pred, dim=1)
                    y_pred = y_pred.cpu().detach().numpy()

                    mask = mask.cpu().detach().numpy()
                    bs = y_pred.shape[0]

                    curr_iou = []
                    for target_num in ids:
                        y_pred_tmp = (y_pred == target_num).astype(int)
                        mask_tmp = (mask == target_num).astype(int)

                        intersection = (y_pred_tmp & mask_tmp).sum()
                        union = (y_pred_tmp | mask_tmp).sum()

                        unions[target_num] += union
                        intersections[target_num] += intersection

                        if not union == 0:
                            curr_iou.append(intersection / union)
                mean_ious = []
                for target_num in ids:
                    mean_ious.append(intersections[target_num] / (1e-8 + unions[target_num]))
                mean_iou_val = np.array(mean_ious).mean()

                if mean_iou_val > best_val_miou:
                    best_val_miou = mean_iou_val
                    unions = {}
                    intersections = {}
                    for target_num in ids:
                        unions[target_num] = 0
                        intersections[target_num] = 0

                    with torch.no_grad():
                        for _, da, in enumerate(test_data):
                            img, mask = da[0], da[1]
                            if img.size(1) == 4:
                                img = img[:, :-1, :, :]

                            img = img.cuda()
                            mask = mask.cuda()
                            input_img_tensor = []
                            for b in range(img.size(0)):
                                input_img_tensor.append(resnet_transform(img[b]))
                            input_img_tensor = torch.stack(input_img_tensor)

                            y_pred = classifier(input_img_tensor)#['out']
                            y_pred = torch.log_softmax(y_pred, dim=1)
                            _, y_pred = torch.max(y_pred, dim=1)
                            y_pred = y_pred.cpu().detach().numpy()
                            mask = mask.cpu().detach().numpy()
                            curr_iou = []
                            for target_num in ids:
                                y_pred_tmp = (y_pred == target_num).astype(int)
                                mask_tmp = (mask == target_num).astype(int)

                                intersection = (y_pred_tmp & mask_tmp).sum()
                                union = (y_pred_tmp | mask_tmp).sum()

                                unions[target_num] += union
                                intersections[target_num] += intersection

                                if not union == 0:
                                    curr_iou.append(intersection / union)

                            img = img.cpu().numpy()
                            img =  img * 255.
                            img = np.transpose(img, (0, 2, 3, 1)).astype(np.uint8)

                        test_mean_ious = []

                        for target_num in ids:
                            test_mean_ious.append(intersections[target_num] / (1e-8 + unions[target_num]))
                        best_test_miou = np.array(test_mean_ious).mean()

                        print("Best IOU ,", str(best_test_miou), "CP: ", resume)

        cross_mIOU.append(best_test_miou)

    print(cross_mIOU)
    print(" cross validation mean:" , np.mean(cross_mIOU) )
    print(" cross validation std:", np.std(cross_mIOU))
    result = {"Cross validation mean": np.mean(cross_mIOU), "Cross validation std": np.std(cross_mIOU), "Cross validation":cross_mIOU }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str,  default="")
    parser.add_argument('--data_images', metavar='DIR',
                    help='path to images')
    parser.add_argument('--data_parts', metavar='DIR',
                    help='path to parts')
    args = parser.parse_args()
    cross_validate(args.resume, args)

