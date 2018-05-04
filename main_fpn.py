import numpy as np
import os, sys
from constants import *
from model_fpn import I2D
import argparse, time
from utils.net_utils import adjust_learning_rate
import torch
from torch.autograd import Variable
# from dataset.dataloader import DepthDataset
from dataset.nyuv2_dataset import NYUv2Dataset
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data.sampler import Sampler
from collections import Counter
import matplotlib, cv2
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class RMSE_log(nn.Module):
    def __init__(self):
        super(RMSE_log, self).__init__()
    
    def forward(self, fake, real):
        if not fake.shape == real.shape:
            _,_,H,W = real.shape
            fake = F.upsample(fake, size=(H,W), mode='bilinear')
        loss = torch.sqrt( torch.mean( torch.abs(torch.log(real)-torch.log(fake)) ** 2 ) )
        return loss

class L1(nn.Module):
    def __init__(self):
        super(L1, self).__init__()
    
    def forward(self, fake, real):
        if not fake.shape == real.shape:
            _,_,H,W = real.shape
            fake = F.upsample(fake, size=(H,W), mode='bilinear')
        loss = torch.mean( torch.abs(10.*real-10.*fake) )
        return loss

class L1_log(nn.Module):
    def __init__(self):
        super(L1_log, self).__init__()
    
    def forward(self, fake, real):
        if not fake.shape == real.shape:
            _,_,H,W = real.shape
            fake = F.upsample(fake, size=(H,W), mode='bilinear')
        loss = torch.mean( torch.abs(torch.log(real)-torch.log(fake)) )
        return loss
    
class BerHu(nn.Module):
    def __init__(self, threshold=0.2):
        super(BerHu, self).__init__()
        self.threshold = threshold
    
    def forward(real, fake):
        mask = real>0
        if not fake.shape == real.shape:
            _,_,H,W = real.shape
            fake = F.upsample(fake, size=(H,W), mode='bilinear')
        fake = fake * mask
        diff = torch.abs(real-fake)
        delta = self.threshold * torch.max(diff).data.cpu().numpy()[0]

        part1 = -F.threshold(-diff, -delta, 0.)
        part2 = F.threshold(diff**2 - delta**2, 0., -delta**2.) + delta**2
        part2 = part2 / (2.*delta)

        loss = part1 + part2
        loss = torch.sum(loss)
        return loss
    
class RMSE(nn.Module):
    def __init__(self):
        super(RMSE, self).__init__()
    
    def forward(self, fake, real):
        if not fake.shape == real.shape:
            _,_,H,W = real.shape
            fake = F.upsample(fake, size=(H,W), mode='bilinear')
        loss = torch.sqrt( torch.mean( torch.abs(10.*real-10.*fake) ** 2 ) )
        return loss

class GradLoss(nn.Module):
    def __init__(self):
        super(GradLoss, self).__init__()
    
    # L1 norm
    def forward(self, grad_fake, grad_real):
        
        return torch.sum( torch.mean( torch.abs(grad_real-grad_fake) ) )

    
class NormalLoss(nn.Module):
    def __init__(self):
        super(NormalLoss, self).__init__()
    
    def forward(self, grad_fake, grad_real):
        prod = ( grad_fake[:,:,None,:] @ grad_real[:,:,:,None] ).squeeze(-1).squeeze(-1)
        fake_norm = torch.sqrt( torch.sum( grad_fake**2, dim=-1 ) )
        real_norm = torch.sqrt( torch.sum( grad_real**2, dim=-1 ) )
        
        return 1 - torch.mean( prod/(fake_norm*real_norm) )
            
# def get_acc(output, target):
#     # takes in two tensors to compute accuracy
#     pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
#     correct = pred.eq(target.data.view_as(pred)).cpu().sum()
#     print("Target: ", Counter(target.data.cpu().numpy()))
#     print("Pred: ", Counter(pred.cpu().numpy().flatten().tolist()))
#     return float(correct)*100 / target.size(0) 


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Single image depth estimation')
    parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='nyuv2', type=str)
    parser.add_argument('--epochs', dest='max_epochs',
                      help='number of epochs to train',
                      default=NUM_EPOCHS, type=int)
    parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
    parser.add_argument('--bs', dest='bs',
                      help='batch_size',
                      default=16, type=int)
    parser.add_argument('--num_workers', dest='num_workers',
                      help='num_workers',
                      default=1, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval',
                      help='display interval',
                      default=10, type=int)
    parser.add_argument('--output_dir', dest='output_dir',
                      help='output directory',
                      default='saved_models', type=str)

# config optimization
    parser.add_argument('--o', dest='optimizer',
                      help='training optimizer',
                      default="sgd", type=str)
    parser.add_argument('--lr', dest='lr',
                      help='starting learning rate',
                      default=1e-3, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is epoch',
                      default=5, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=0.1, type=float)

# set training session
    parser.add_argument('--s', dest='session',
                      help='training session',
                      default=1, type=int)
    parser.add_argument('--eval_epoch', dest='eval_epoch',
                      help='number of epoch to evaluate',
                      default=2, type=int)

# resume trained model
    parser.add_argument('--r', dest='resume',
                      help='resume checkpoint or not',
                      default=False, type=bool)
    parser.add_argument('--start_at', dest='start_epoch',
                      help='epoch to start with',
                      default=0, type=int)
    parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load model',
                      default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load model',
                      default=0, type=int)

# training parameters
    parser.add_argument('--gamma_sup', dest='gamma_sup',
                      help='factor of supervised loss',
                      default=1., type=float)
    parser.add_argument('--gamma_unsup', dest='gamma_unsup',
                      help='factor of unsupervised loss',
                      default=1., type=float)
    parser.add_argument('--gamma_reg', dest='gamma_reg',
                      help='factor of regularization loss',
                      default=10., type=float)

    args = parser.parse_args()
    return args

def get_coords(b, h, w):
    i_range = Variable(torch.arange(0, h).view(1, h, 1).expand(b,1,h,w))  # [B, 1, H, W]
    j_range = Variable(torch.arange(0, w).view(1, 1, w).expand(b,1,h,w))  # [B, 1, H, W]
    coords = torch.cat((j_range, i_range), dim=1)
    norm = Variable(torch.Tensor([w,h]).view(1,2,1,1))
    coords = coords * 2. / norm - 1.
    coords = coords.permute(0, 2, 3, 1)
   
    return coords
        
def resize_tensor(img, coords):
    return nn.functional.grid_sample(img, coords, mode='bilinear', padding_mode='zeros')

def imgrad(img):
    img = torch.mean(img, 1, True)
    fx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fx).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.cuda()
    conv1.weight = nn.Parameter(weight)
    grad_x = conv1(img)

    fy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fy).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.cuda()
    conv2.weight = nn.Parameter(weight)
    grad_y = conv2(img)

#     grad = torch.sqrt(torch.pow(grad_x,2) + torch.pow(grad_y,2))
    
    return grad_y, grad_x

def imgrad_yx(img):
    N,C,_,_ = img.size()
    grad_y, grad_x = imgrad(img)
    return torch.cat((grad_y.view(N,C,-1), grad_x.view(N,C,-1)), dim=1)

def reg_scalor(grad_yx):
    return torch.exp(-torch.abs(grad_yx)/255.)
    
    
class sampler(Sampler):
  def __init__(self, train_size, batch_size):
    self.num_data = train_size
    self.num_per_batch = int(train_size / batch_size)
    self.batch_size = batch_size
    self.range = torch.arange(0,batch_size).view(1, batch_size).long()
    self.leftover_flag = False
    if train_size % batch_size:
      self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
      self.leftover_flag = True

  def __iter__(self):
    rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
    self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

    self.rand_num_view = self.rand_num.view(-1)

    if self.leftover_flag:
      self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

    return iter(self.rand_num_view)

  def __len__(self):
    return self.num_data

def collate_fn(data):
    imgs, depths = zip(*data)
    B = len(imgs)
    im_batch = torch.ones((B,3,376,1242))
    d_batch = torch.ones((B,1,376,1242))
    for ind in range(B):
        im, depth = imgs[ind], depths[ind]
        im_batch[ind, :, -im.shape[1]:, :im.shape[2]] = im
        d_batch[ind, :, -depth.shape[1]:, :depth.shape[2]] = depth
    return im_batch, d_batch

if __name__ == '__main__':

    args = parse_args()

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You might want to run with --cuda")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    # dataset
    if args.dataset == 'kitti':
        train_dataset = DepthDataset(root='/disk2/depth_data/kitti/train') # KittiDataset(train=True)
        eval_dataset = DepthDataset(root='/disk2/depth_data/kitti/train') # KittiDataset(train=False)
#         train_dataset = DepthDataset(root='../data/kitti/train') # KittiDataset(train=True)
#         eval_dataset = DepthDataset(root='../data/kitti/train') # KittiDataset(train=False)
        train_size = len(train_dataset)
        eval_size = len(eval_dataset)
        print(train_size, eval_size)

        train_batch_sampler = sampler(train_size, args.bs)
        eval_batch_sampler = sampler(eval_size, args.bs)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs,
                                shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers)
        
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.bs,
                                shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers)
        
    elif args.dataset == 'nyuv2':
        train_dataset = NYUv2Dataset()
        train_size = len(train_dataset)
        eval_dataset = NYUv2Dataset(train=False)
        eval_size = len(eval_dataset)
        print(train_size)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs,
                                shuffle=True, num_workers=args.num_workers)
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.bs,
                                shuffle=True, num_workers=args.num_workers)
        
    elif args.dataset == 'scannet':
        pass

    # network initialization
    print('Initializing model...')
    i2d = I2D(fixed_feature_weights=False)
    if args.cuda:
        i2d = i2d.cuda()
        
    print('Done!')

    # hyperparams
    lr = args.lr
    bs = args.bs
    lr_decay_step = args.lr_decay_step
    lr_decay_gamma = args.lr_decay_gamma

    # params
    params = []
    for key, value in dict(i2d.named_parameters()).items():
      if value.requires_grad:
        if 'bias' in key:
          params += [{'params':[value],'lr':lr*(DOUBLE_BIAS + 1), \
                  'weight_decay': 4e-5 and WEIGHT_DECAY or 0}]
        else:
          params += [{'params':[value],'lr':lr, 'weight_decay': 4e-5}]

    # optimizer
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=4e-5)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)

    rmse = RMSE()
    depth_criterion = RMSE_log()
    grad_criterion = GradLoss()
    normal_criterion = NormalLoss()
    eval_metric = RMSE_log()
    
    # resume
    if args.resume:
        load_name = os.path.join(args.output_dir,
          'i2d_1_{}.pth'.format(args.checkepoch))
        print("loading checkpoint %s" % (load_name))
        state = i2d.state_dict()
        checkpoint = torch.load(load_name)
        args.start_epoch = checkpoint['epoch']
        checkpoint = {k: v for k, v in checkpoint['model'].items() if k in state}
        state.update(checkpoint)
        i2d.load_state_dict(state)
#         optimizer.load_state_dict(checkpoint['optimizer'])
#         lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (load_name))
        del checkpoint
        torch.cuda.empty_cache()

    # constants
    iters_per_epoch = int(train_size / args.bs)
    
    grad_factor = 10.
    normal_factor = 1.
    
    for epoch in range(args.start_epoch, args.max_epochs):
        
        # setting to train mode
        i2d.train()
        start = time.time()
        if epoch % (args.lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma

        img = Variable(torch.FloatTensor(1))
        z = Variable(torch.FloatTensor(1))
        if args.cuda:
            img = img.cuda()
            z = z.cuda()
        
        train_data_iter = iter(train_dataloader)
        
        for step in range(iters_per_epoch):
            start = time.time()
            data = train_data_iter.next()
            
            img.data.resize_(data[0].size()).copy_(data[0])
            z.data.resize_(data[1].size()).copy_(data[1])

            optimizer.zero_grad()
            z_fake = i2d(img)
            depth_loss = depth_criterion(z_fake, z)
            
            grad_real, grad_fake = imgrad_yx(z), imgrad_yx(z_fake)
            grad_loss = grad_criterion(grad_fake, grad_real)     * grad_factor * (epoch>3)
            normal_loss = normal_criterion(grad_fake, grad_real) * normal_factor * (epoch>7)
            
            loss = depth_loss + grad_loss + normal_loss
            loss.backward()
            optimizer.step()

            end = time.time()

            # info
            if step % args.disp_interval == 0:

                print("[epoch %2d][iter %4d] loss: %.4f RMSElog: %.4f grad_loss: %.4f normal_loss: %.4f" \
                                % (epoch, step, loss, depth_loss, grad_loss, normal_loss))
#                 print("[epoch %2d][iter %4d] loss: %.4f iRMSE: %.4f" \
#                                 % (epoch, step, loss, metric))
        # save model
        save_name = os.path.join(args.output_dir, 'i2d_{}_{}.pth'.format(args.session, epoch))

        torch.save({'epoch': epoch+1,
                    'model': i2d.state_dict(), 
#                     'optimizer': optimizer.state_dict(),
                   },
                   save_name)

        print('save model: {}'.format(save_name))
        print('time elapsed: %fs' % (end - start))
            
        if epoch % 1 == 0:
            # setting to eval mode
            i2d.eval()

            img = Variable(torch.FloatTensor(1), volatile=True)
            z = Variable(torch.FloatTensor(1), volatile=True)
            if args.cuda:
                img = img.cuda()
                z = z.cuda()

            print('evaluating...')

            eval_loss = 0
            rmse_accum = 0
            count = 0
            eval_data_iter = iter(eval_dataloader)
            for i, data in enumerate(eval_data_iter):
                print(i,'/',len(eval_data_iter)-1)

                img.data.resize_(data[0].size()).copy_(data[0])
                z.data.resize_(data[1].size()).copy_(data[1])

                z_fake = i2d(img)
                depth_loss = float(img.size(0)) * rmse(z_fake, z)**2
                eval_loss += depth_loss
                rmse_accum += float(img.size(0)) * eval_metric(z_fake, z)**2
                count += float(img.size(0))

            print("[epoch %2d] RMSE_log: %.4f RMSE: %.4f" \
                            % (epoch, torch.sqrt(eval_loss/count), torch.sqrt(rmse_accum/count)))
            with open('val.txt', 'a') as f:
                f.write("[epoch %2d] RMSE_log: %.4f RMSE: %.4f\n" \
                            % (epoch, torch.sqrt(eval_loss/count), torch.sqrt(rmse_accum/count)))

       

