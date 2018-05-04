import torch.utils.data as data
import numpy as np
from PIL import Image
from scipy.misc import imread
from path import Path
from constants import *
from torchvision.transforms import Resize, Compose, ToPILImage, ToTensor, RandomHorizontalFlip
import torch, time
import torch.nn.functional as F

def load_depth(filename):
    depth_png = np.asarray(Image.open(filename))

    assert(np.max(depth_png) > 255)

    depth = depth_png.astype(np.float32) / (256. * 256.)
    
    depth[depth == 0] = 1.
    
    return depth

class KittiDataset(data.Dataset):
    def __init__(self, root='/disk2/data/', seed=None, train=True):
        
        np.random.seed(seed)
        self.root = Path(root)
        img_dir = self.root/'kitti_train_images.txt' if train else self.root/'kitti_val_images.txt'
        depth_dir = self.root/'kitti_train_depth_maps.txt' if train else self.root/'kitti_val_depth_maps.txt'
        # intr_dir = self.root/'kitti_train_intrinsics.txt' if train else self.root/'kitti_val_intrinsics.txt'
        self.img_l_paths = [d[:-1] for d in open(img_dir) if 'image_02' in d]
        self.depth_l_paths = [d[:-1] for d in open(depth_dir) if 'image_02' in d]
        
        # at least 20 frames between 2 examples
        del_idxs = []
        cur = 0
        for i in range(1,len(self.img_l_paths)):
            idx = int(self.img_l_paths[i][-7:-4])
            cur_idx = int(self.img_l_paths[cur][-7:-4])
            if abs(idx-cur_idx) < 3:
                del_idxs += [i]
            else:
                cur = i
        self.img_l_paths = np.delete(self.img_l_paths, del_idxs)
        self.depth_l_paths = np.delete(self.depth_l_paths, del_idxs)

        self.length = len(self.img_l_paths)
            
    def __getitem__(self, index):
        depth = torch.FloatTensor( load_depth(self.depth_l_paths[index])[None,:,:] )
        img = ToTensor()( Image.open(self.img_l_paths[index]) )
        
        tpad = 376 - img.size(1) 
        rpad = 1242 - img.size(2)
        
        # (padLeft, padRight, padTop, padBottom)
        img = F.pad(img.unsqueeze(0), pad=(0, rpad, tpad, 0), mode='reflect')
        depth = F.pad(depth.unsqueeze(0), pad=(0, rpad, tpad, 0), mode='constant', value=1.)
        return img.data.squeeze(0), depth.data.squeeze(0)

    def __len__(self):
#         return 16 # for debug purpose
        return self.length

if __name__ == '__main__':
    # Testing
    dataset = KittiDataset()
    print(len(dataset))
    for item in dataset[0]:
        print(item.size())
