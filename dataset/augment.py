import torch.utils.data as data
import numpy as np
from PIL import Image
from scipy.misc import imread
from path import Path
from constants import *
from torchvision.transforms import Resize, Compose, ToPILImage, ToTensor, RandomHorizontalFlip, CenterCrop, ColorJitter
import torch, time, os
import torch.nn.functional as F
import random
import scipy.ndimage as ndimage
from scipy import misc

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w, :]

        landmarks = landmarks[top: top + new_h,
                      left: left + new_w]

        return image, landmarks


class CropCenter(object):
    """Crops the given inputs and target arrays at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    Careful, img1 and img2 may not be the same size
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, inputs, target):
        h1, w1, _ = inputs.shape
        th, tw = self.size
        x1 = int(round((w1 - tw) / 2.))
        y1 = int(round((h1 - th) / 2.))

        inputs = inputs[y1 : y1 + th, x1 : x1 + tw]
        target = target[y1 : y1 + th, x1 : x1 + tw]
        return inputs,target
    
class RandomCropRotate(object):
    """Random rotation of the image from -angle to angle (in degrees)
    A crop is done to keep same image ratio, and no black pixels
    angle: max angle of the rotation, cannot be more than 180 degrees
    interpolation order: Default: 2 (bilinear)
    """
    def __init__(self, angle, diff_angle=0, order=2):
        self.angle = angle
        self.order = order
        self.diff_angle = diff_angle

    def __call__(self, sample):
        inputs,target = sample
        h,w,_ = inputs.shape

        applied_angle  = random.uniform(-self.angle,self.angle)
        diff = random.uniform(-self.diff_angle,self.diff_angle)
        angle1 = applied_angle - diff/2

        angle1_rad = angle1*np.pi/180
        
        inputs = ndimage.interpolation.rotate(inputs, angle1, reshape=True, order=self.order)
        target = ndimage.interpolation.rotate(target, angle1, reshape=True, order=self.order)
        
        #keep angle1 and angle2 within [0,pi/2] with a reflection at pi/2: -1rad is 1rad, 2rad is pi - 2 rad
        angle1_rad = np.pi/2 - np.abs(angle1_rad%np.pi - np.pi/2)
        
        c1 = np.cos(angle1_rad)
        s1 = np.sin(angle1_rad)
        c_diag = h/np.sqrt(h*h+w*w)
        s_diag = w/np.sqrt(h*h+w*w)

        ratio = 1./(c1+w/float(h)*s1)

        crop = CropCenter((int(h*ratio),int(w*ratio)))
        return crop(inputs, target)
    
class RandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image, landmarks = sample
        if np.random.randn() > 0.5:
            image = image[:,::-1,:]
            landmarks = landmarks[:,::-1]

        return image, landmarks
    
if __name__ == '__main__':
    root = '/disk2/data/nyuv2/'
    rgb_paths = [root+'train_rgb/'+d for d in os.listdir(root+'train_rgb/')]
    augmentation = Compose([RandomCrop((420,560)), RandomCropRotate(10)])
    counter = 3
    while counter < 6:
        counter += 1
        c = 0
        for path in rgb_paths:
            c += 1
            print('%d %d/%d' %(counter, c, len(rgb_paths)))
            img = Image.open(path)
            depth = Image.open(path.replace('rgb', 'depth'))
            rgb, depth = np.array(img), np.array(depth)[:,:,None]
            rgb, depth = augmentation((rgb, depth))
            depth = depth.squeeze(-1)
            rgb, depth = Image.fromarray(rgb), Image.fromarray(depth)
            rgb, depth = Resize((420,560))(rgb), Resize((420,560))(depth)
            rgb = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)(rgb)
            
            rgb.save(root+'aug_rgb/'+path.split('/')[-1].split('.')[0]+str(counter)+'.png')
            
            depth.save(root+'aug_depth/'+path.replace('rgb', 'depth').split('/')[-1].split('.')[0]+str(counter)+'.png')
            