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
from utils.evaluation_metrics import *

def valid(args, device, valid_loader, model, epoch):
    '''
    description: valid one epoch of the model 
    parameter: args, device, train_loader, modelr, epoch
    return: empty
    '''
    model.eval()
    for i, (image, layout_depth, layout_seg, intrinsic) in enumerate(valid_loader):
        start = time.time()

        image = image.to(device)
        layout_depth = layout_depth.to(device)
        layout_seg = layout_seg.to(device)
        intrinsic = intrinsic.to(device)


        parameter = model(image)

        max_num = get_plane_max_num(layout_seg)
        average_plane_info = get_average_plane_info(device, parameter, layout_seg, max_num)
        parameter_gt = get_parameter(device, layout_depth, layout_seg)
        average_depth = get_average_depth_map(device, layout_seg, average_plane_info)

        loss = parameter_loss(parameter, parameter_gt) + \
            depth_loss(average_depth, layout_depth) + \
            discrimitive_loss(parameter, layout_seg, average_plane_info, args.delta_v, args.delta_d)
        
        depth_mine = get_depth_map(device, parameter)
        rms, rel, rlog10, rate_1, rate_2, rate_3 = depth_metrics(depth_mine, layout_depth)
        end = time.time()
        the_time = end - start

        result_string = ('Valid: Epoch: [{} / {}], Batch: [{} / {}], Time: {:.3f}s, Loss: {:.4f}\n' \
            + 'rms: {:.3f}, rel: {:.3f}, log10: {:.3f}, delta1: {:.3f}, delta2: {:.3f}, delta3: {:.3f}') \
            .format(epoch + 1, args.epochs, i + 1, len(valid_loader), the_time, loss.item(), \
                rms, rel, rlog10, rate_1, rate_2, rate_3)
        write_log(args, epoch, i, 'validation', result_string)
        print(result_string)
 