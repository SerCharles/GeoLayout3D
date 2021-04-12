''' 
The function of training one epoch of the network
'''

import numpy as np
import time
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm

from data.dataset import *
import models.senet as senet 
import models.modules as modules 
import models.net as net
from utils.loss import * 
from global_utils import *
from utils.utils import *

def train(args, device, train_loader, model, optimizer, epoch):
    '''
    description: train one epoch of the model 
    parameter: args, device, train_loader, model, optimizer, epoch
    return: the model trained
    '''
    model.train()
    #print(train_loader)
    adjust_learning_rate(args, optimizer, epoch)
    for i, (image, layout_depth, layout_seg, intrinsic) in enumerate(train_loader):

        
        start = time.time()
        image = image.to(device)
        layout_depth = layout_depth.to(device)
        layout_seg = layout_seg.to(device)
        intrinsic = intrinsic.to(device)
        

        optimizer.zero_grad()
        parameter = model(image)

        max_num = get_plane_max_num(layout_seg)
        average_plane_info = get_average_plane_info(device, parameter, layout_seg, max_num)
        parameter_gt = get_parameter(device, layout_depth, layout_seg, args.epsilon)
        parameter_gt = parameter_gt.detach()
        #average_depth = get_average_depth_map(device, layout_seg, average_plane_info, args.epsilon)

        
        loss_p = parameter_loss(parameter, parameter_gt)
        loss_dis = discrimitive_loss(parameter, layout_seg, average_plane_info, args.delta_v, args.delta_d) * args.alpha
        loss_d = depth_loss_direct(device, layout_seg, average_plane_info, layout_depth, args.epsilon)  * args.beta
        loss = loss_p + loss_dis + loss_d
        loss.backward()
        optimizer.step()

        end = time.time()
        the_time = end - start

        result_string = 'Train: Epoch: [{} / {}], Batch: [{} / {}], Time {:.3f}s, \n' \
        .format(epoch + 1, args.epochs, i + 1, len(train_loader), the_time) + \
        'Loss {:.4f}, Loss Parameter {:.4f}, Loss Discrimitive {:.4f}, Loss Depth {:.4f}' \
            .format(loss.item(), loss_p.item(), loss_dis.item(), loss_d.item())
        print(result_string)
        write_log(args, epoch, i, 'training', result_string)
        
    return model
 