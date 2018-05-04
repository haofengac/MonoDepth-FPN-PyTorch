import torch.utils.data as data
import numpy as np
from PIL import Image
from scipy.misc import imread
# from path import Path
import os
from glob import glob
from constants import *
from torchvision.transforms import Resize, Compose, ToPILImage, ToTensor #, RandomHorizontalFlip
import torch, time
import torch.nn.functional as F

class KittiDataset(data.Dataset):
    def __init__(self, train=True):
        root = '/disk2/data/eigen'
#         root = '/media/xander/Essence/kitti'
        if train:
            self.root = os.path.join(root, 'train')
        else:
#             self.root = os.path.join(root, 'val')
            self.root = os.path.join(root, 'test')
        self.path_temp = os.path.join(self.root, '%s', '%s.%s')
        self.imtype = 'png'
        if train:
            self.length = len(glob(self.path_temp%('depth_out', '*', self.imtype)))
            self.files = glob(self.path_temp%('depth_out', '*', self.imtype))
        else:
            self.length = len(glob(self.path_temp%('depth_out', '*', self.imtype)))
            self.files = glob(self.path_temp%('depth_out', '*', self.imtype))
        self.train = train
        self.rgb_transform = Compose([Resize([192,640]), ToTensor()])
        self.depth_transform = Compose([Resize([48,160]), ToTensor()])
        
    def __getitem__(self, index):
        
        if self.train:
            depth = Image.open( self.files[index] )
            img = Image.open( self.files[index].replace('depth_out', 'img') )
            rgb, d = self.rgb_transform(img), self.depth_transform(depth)
        else:
            depth = Image.open( self.files[index] )
            img = Image.open( self.files[index].replace('depth_out', 'img') )
            rgb, d = self.rgb_transform(img), self.depth_transform(depth)
        rgb, d = rgb[:,64:,:], d[:,16:,:]
        
        return rgb, d.float()/65536.

    def __len__(self):
#         return 16 # for debug purpose
        return self.length

class ScannetDataset(data.Dataset):
    def __init__(self, train=True):
        if train:
            self.root = os.path.join('/disk2/data/scannet', 'train')
        else:
            self.root = os.path.join('/disk2/data/scannet', 'val')
        self.path_temp = os.path.join(self.root, '%s', '%s.%s')
        self.imtype = 'jpg'
        self.length = len(glob(self.path_temp%('img', '*', self.imtype)))
        self.train = train
        self.rgb_transform = Compose([Resize([484, 648]), ToTensor()])
        self.depth_transform = Compose([Resize([121, 162]), ToTensor()])
        
    def __getitem__(self, index):
        
        img = Image.open( self.path_temp%('img',str(index).zfill(5),self.imtype) )
  
        if self.train:
            depth = Image.open( self.path_temp%('depth_out',str(index).zfill(5),'png') )
            img, depth = self.rgb_transform(img), self.depth_transform(depth)
        else:
            depth = Image.open( self.path_temp%('depth',str(index).zfill(5),'png') )
            img, depth = ToTensor()(img), ToTensor()(depth)

        return img, depth.float()/65536.

    def __len__(self):
#         return 16 # for debug purpose
        return self.length


kitti_train = KittiDataset()
kitti_val = KittiDataset(train=False)
scannet_train = ScannetDataset()
scannet_val = ScannetDataset(train=False)

if __name__ == '__main__':
    # Testing
    for item in kitti_train[0]:
        print(item.size())
        
    for item in scannet_train[0]:
        print(item.size())
