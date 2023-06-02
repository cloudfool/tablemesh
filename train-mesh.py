import os
import torch
#import torch.utils.data as data
#import numpy as np
#import scipy.io as scio
import argparse
import time
#import torch.nn as nn
#import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import random
#import h5py
#import re
#import water
#from test import test


#from math import exp
from data_loader import ICDAR2015, Synth80k, ICDAR2013, Receipt, Pdf,TableMesh

###import file#######
#from augmentation import random_rot, crop_img_bboxes
#from gaussianmap import gaussion_transform, four_point_transform
#from generateheatmap import add_character, generate_target, add_affinity, generate_affinity, sort_box, real_affinity, generate_affinity_box
from mseloss import Maploss,boxloss,meshloss
from collections import OrderedDict
#from eval.script import getresult



#from PIL import Image
#from torchvision.transforms import transforms
from dtgs import DTGS_1
from torch.autograd import Variable
#from multiprocessing import Pool
#from BalancedDataParallel import BalancedDataParallel

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#3.2768e-5
random.seed(42)

# class SynAnnotationTransform(object):
#     def __init__(self):
#         pass
#     def __call__(self, gt):
#         image_name = gt['imnames'][0]
parser = argparse.ArgumentParser(description='CRAFT reimplementation')


parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--batch_size', default=128, type = int,
                    help='batch size of training')
#parser.add_argument('--cdua', default=True, type=str2bool,
                    #help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=3.2768e-5, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--num_workers', default=1, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--local_rank', default=0, type=int,
                    help='Number of workers used in dataloading')

args = parser.parse_args()

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (0.8 ** step)
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':



    net = DTGS_1()

    #net.load_state_dict(copyStateDict(torch.load('./pretrain/craft_mlt_25k.pth')))
    #net.load_state_dict(copyStateDict(torch.load('./pretrain/91.pth')))

    #net = net.cuda()


    #net = torch.nn.DataParallel(net,device_ids=[0,1]).cuda()
    net = torch.nn.DataParallel(net,device_ids=[0]).cuda()
    #net = BalancedDataParallel(1,net,dim=0).cuda()
    #torch.distributed.init_process_group(backend="nccl")
    #net = torch.nn.parallel.DistributedDataParallel(net,device_ids=[0,1,2]).cuda()

    cudnn.benchmark = True
    #realdata = TableMesh(net, './data/xhqd_tablecell', target_size=1024)
    #realdata = TableMesh(net, './data/ICDAR2013_table', target_size=512)
    #realdata = TableMesh(net, './data/PubTabNet', target_size=1024)
    realdata = TableMesh( './data/TableSplerge_line', target_size=1024)
    #print (len(realdata))

    real_data_loader = torch.utils.data.DataLoader(
        realdata,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=True)

    #dataloader = TableMesh( './data/WTW', target_size = 1024 )
    #train_loader = torch.utils.data.DataLoader(
    #    dataloader,
    #    batch_size=1,
    #    shuffle=True,
    #    num_workers=0,
    #    drop_last=True,
    #    pin_memory=True)
    #batch_syn = iter(train_loader)


    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = meshloss()
    #criterion = torch.nn.MSELoss(reduce=True, size_average=True)
    #net.train()
    step_index = 0
    loss_time = 0
    loss_value = 0
    compare_loss = 1
    for epoch in range(1000):
        train_time_st = time.time()
        loss_value = 0
        if epoch % 50 == 0 and epoch != 0:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        st = time.time()
        for index, (real_images, real_gah_label1, real_gah_label2, real_mask, _) in enumerate(real_data_loader):

            # try:
            #     syn_images, syn_gah_label1, syn_gah_label2, syn_mask, __ = next(batch_syn)
            # except:
            #     batch_syn = iter(train_loader)
            #     syn_images, syn_gah_label1, syn_gah_label2, syn_mask, __ = next(batch_syn)


            # images = torch.cat((syn_images,real_images), 0)
            # gah_label1 = torch.cat((syn_gah_label1, real_gah_label1), 0)
            # gah_label2 = torch.cat((syn_gah_label2, real_gah_label2), 0)


            net.train()
            #real_images, real_gh_label, real_gah_label, real_mask = next(batch_real)
            #affinity_mask = torch.cat((syn_mask, real_affinity_mask), 0)

            images = real_images
            gah_label1 = real_gah_label1
            gah_label2 = real_gah_label2
            #mask = real_mask


            images = Variable(images.type(torch.FloatTensor)).cuda()
            gah_label1 = gah_label1.type(torch.FloatTensor)
            gah_label1 = Variable(gah_label1).cuda()
            gah_label2 = gah_label2.type(torch.FloatTensor)
            gah_label2 = Variable(gah_label2).cuda()
            #mask = mask.type(torch.FloatTensor)
            #mask = Variable(mask).cuda()


            out, _ = net(images)

            optimizer.zero_grad()

            out1 = out[:, :, :, 0].cuda()
            out2 = out[:, :, :, 1].cuda()
            loss = criterion(gah_label1, gah_label2, out1, out2, None)
            #loss = criterion(gah_label,out,mask)
            #print ('loss',loss.item())

            loss.backward()
            optimizer.step()
            loss_value += loss.item()
            if index % 2 == 0 and index > 0:
                et = time.time()
                print('epoch {}:({}/{}) batch || training time for 2 batch {} || training loss {} ||'.format(epoch, index, len(real_data_loader), et-st, loss_value/2))
                loss_time = 0
                loss_value = 0
                st = time.time()
            # if loss < compare_loss:
            #     print('save the lower loss iter, loss:',loss)
            #     compare_loss = loss
            #     torch.save(net.module.state_dict(),
            #                '/data/CRAFT-pytorch/real_weights/lower_loss.pth')

            net.eval()
            #if index>10:
            #    break
        if epoch%1 == 0:
            print('Saving state, iter:', epoch)
            torch.save(net.module.state_dict(),
                   './weights/ppm4_line_20230602' + repr(epoch) + '.pth')
        #test('./weights/CRAFT_clr_' + repr(epoch) + '.pth')
        #test('/data/CRAFT-pytorch/craft_mlt_25k.pth')
        #getresult()









