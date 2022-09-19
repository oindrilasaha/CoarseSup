import torch.utils.data as data
from torch import from_numpy

from PIL import Image
import os
import os.path
import h5py
import numpy as np
import cv2
import torchfile
from utils import vis_square
import torch

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.npy']

def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        # img = np.load(f)
        # return img
        img = Image.open(f)
        return img #.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

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

list_ex = ['Ivory_Gull_0085_49456.jpg', 'Ivory_Gull_0040_49180.jpg', 'Mallard_0130_76836.jpg', 'White_Necked_Raven_0070_102645.jpg', 'Clark_Nutcracker_0020_85099.jpg', 'Pelagic_Cormorant_0022_23802.jpg', 'Western_Gull_0002_54825.jpg', 'Brewer_Blackbird_0028_2682.jpg', 'Black_Tern_0079_143998.jpg']

class Heatmap(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, images_root, masks_root, mode=True, transform=None, target_transform=None,
                 loader=default_loader):
        # imgs = make_dataset(root)
        # if len(imgs) == 0:
        #     raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
        #                        "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.images_root = images_root
        self.masks_root = masks_root
        # self.imgs = imgs
        self.transform = transform
        self.loader = loader
        if mode==True:
            self.annos = load_train_annos()
        else:
            self.annos = load_val_annos()

        for key in list_ex:
            self.annos.pop(key,None)


        self.kp_num = 15

    def make_gaussian(self, paths, pt):
        (h,w,_) = pt 
        anno = self.annos[os.path.basename(paths)[:-3]+'jpg']
        masks = np.zeros((self.kp_num, h, w), dtype=np.float32)
        for idx in range(self.kp_num):
            if int(anno[idx+1][2]) == 1:
                masks[idx] = gaussian(masks[idx], (int(round(anno[idx+1][0]*w)), int(round(anno[idx+1][1]*h))))

        return masks


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = list(self.annos.keys())[index]
        # path = self.imgs[index]
        img = np.array(Image.open(os.path.join(self.images_root,path)))
        masks = self.make_gaussian(path, img.shape)
        path = list(self.annos.keys())[index][:-3]+'png'
        figure_ground = Image.open(os.path.join(self.masks_root,path)).convert('L')
        
        figure_ground = np.array(figure_ground)
        figure_ground[np.where(figure_ground<204)]=0.0
        figure_ground[np.where(figure_ground>0)]=1.0

        xmin, xmax, ymin, ymax = np.where(figure_ground!=0)[0].min(), np.where(figure_ground!=0)[0].max(), np.where(figure_ground!=0)[1].min(), np.where(figure_ground!=0)[1].max()
        img_new = img[max(0,xmin-20):min(xmax+20,img.shape[0]),max(0,ymin-20):min(ymax+20, img.shape[1])]
        figure_ground_new = figure_ground[max(0,xmin-20):min(xmax+20,img.shape[0]),max(0,ymin-20):min(ymax+20, img.shape[1])]
        masks_new = masks[:,max(0,xmin-20):min(xmax+20,img.shape[0]),max(0,ymin-20):min(ymax+20, img.shape[1])]

        img = Image.fromarray(img_new)
        figure_ground = Image.fromarray(figure_ground_new)
        masks = cv2.resize(masks_new.transpose(1,2,0), (256,256))

        img = img.resize((256,256), resample=Image.BICUBIC)
        figure_ground = figure_ground.resize((256,256), resample=Image.NEAREST)
        figure_ground = np.array(figure_ground)
        figure_ground = np.stack((1.0-figure_ground.copy(), figure_ground), -1)
        figure_ground = torch.from_numpy(figure_ground)
        # print(img.shape)
        

        if self.transform is not None:
            (img, figure_ground, masks) = self.transform((img, figure_ground, masks))

        try:
            a = figure_ground.permute(2,0,1).float()
        except:
            print(path)
            print(figure_ground.shape)
            import pdb;pdb.set_trace()

        return img, figure_ground.permute(2,0,1).float(), masks.transpose(2,0,1).astype(np.float32), path


    def __len__(self):
        return len(list(self.annos.keys()))
