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
    print(train_loader)
    for i, (depth, image, init_label, layout_depth, layout_seg, intrinsic, norm) in enumerate(train_loader):
        start = time.time()

        #depth, image, init_label, layout_depth, layout_seg, face, intrinsic, norm = the_data 
        image = image.to(device)
        layout_depth = layout_depth.to(device)
        layout_seg = layout_seg.to(device)
        intrinsic = intrinsic.to(device)
        face = get_plane_ids(layout_seg)
        

        optimizer.zero_grad()
        print(image.size())
        parameter = model(image)
        print(parameter.size())
        average_plane_info = get_average_plane_info(parameter, layout_seg, face)
        parameter_gt = get_parameter(depth)
        loss = parameter_loss(parameter, parameter_gt) + \
            discrimitive_loss(parameter, layout_seg, face, average_plane_info, args.delta_v, args.delta_d) + \
            depth_loss(parameter, face, average_plane_info, layout_depth)

        loss.backward()
        optimizer.step()

        end = time.time()
        the_time = end - start

        result_string = 'Train: Epoch: [{} / {}] Batch: [{} / {}] Time {:.3f} Loss {:.4f}' \
            .format(epoch + 1, args.epochs, i + 1, len(train_loader), the_time, loss)
        print(result_string)
        write_log(args, epoch, i, 'training', result_string)
    return model
 