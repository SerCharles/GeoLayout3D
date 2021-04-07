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

        with torch.no_grad():
            parameter = model(image)

            max_num = get_plane_max_num(layout_seg)
            average_plane_info = get_average_plane_info(device, parameter, layout_seg, max_num)
            parameter_gt = get_parameter(device, layout_depth, layout_seg, args.epsilon)
            average_depth = get_average_depth_map(device, layout_seg, average_plane_info, args.epsilon)

            loss_p = parameter_loss(parameter, parameter_gt)
            loss_dis = discrimitive_loss(parameter, layout_seg, average_plane_info, args.delta_v, args.delta_d) * args.alpha
            loss_d = depth_loss(average_depth, layout_depth, args.epsilon) * args.beta
            loss = loss_p + loss_dis + loss_d
        
            depth_mine = get_depth_map(device, parameter, args.epsilon)
            rms, rel, rlog10, rate_1, rate_2, rate_3 = depth_metrics(depth_mine, layout_depth)
        end = time.time()
        the_time = end - start

        result_string = 'Valid: Epoch: [{} / {}], Batch: [{} / {}], Time: {:.3f}s, \n' \
        .format(epoch + 1, args.epochs, i + 1, len(valid_loader), the_time) + \
        'Loss: {:.4f}, Loss Parameter {:.4f}, Loss Discrimitive {:.4f}, Loss Depth {:.4f}, \n' \
            .format(loss.item(), loss_p.item(), loss_dis.item(), loss_d.item()) + \
            'rms: {:.3f}, rel: {:.3f}, log10: {:.3f}, delta1: {:.3f}, delta2: {:.3f}, delta3: {:.3f}' \
            .format(rms, rel, rlog10, rate_1, rate_2, rate_3)
        write_log(args, epoch, i, 'validation', result_string)
        print(result_string)
    save_checkpoint(args, model.state_dict(), epoch)
 