import numpy as np
import time
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from data.load_matterport import *
import models.senet as senet 
import models.modules as modules 
import models.net as net
from utils.loss_geolayout import * 
from train_utils import *
from utils.get_parameter_geolayout import *

def train(args, device, train_loader, model, optimizer, epoch):
    '''
    description: train one epoch of the model 
    parameter: args, device, train_loader, model, optimizer, epoch
    return: the model trained
    '''
    model.train()
    #print(train_loader)
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
        parameter_gt = get_parameter(device, layout_depth, layout_seg)
        average_depth = get_average_depth_map(device, layout_seg, average_plane_info)

        '''
        print('parameter:', parameter.requires_grad)
        print('average_plane_info:', average_plane_info.requires_grad)
        print('parameter_gt:', parameter_gt.requires_grad)
        print('average_depth:', average_depth.requires_grad)
        '''
        
        loss = parameter_loss(parameter, parameter_gt) + \
            depth_loss(average_depth, layout_depth) + \
            discrimitive_loss(device, parameter, layout_seg, average_plane_info, args.delta_v, args.delta_d)
            

        loss.backward()
        optimizer.step()

        end = time.time()
        the_time = end - start

        result_string = 'Train: Epoch: [{} / {}], Batch: [{} / {}], Time {:.3f}s, Loss {:.4f}' \
            .format(epoch + 1, args.epochs, i + 1, len(train_loader), the_time, loss.item())
        print(result_string)
        write_log(args, epoch, i, 'training', result_string)
    return model
 