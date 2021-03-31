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
    for i, (depth, image, init_label, layout_depth, layout_seg, intrinsic, norm) in enumerate(valid_loader):
        start = time.time()

        image = image.to(device)
        layout_depth = layout_depth.to(device)
        layout_seg = layout_seg.to(device)
        intrinsic = intrinsic.to(device)
        face = get_plane_ids(layout_seg)


        parameter = model(image)
        average_plane_info = get_average_plane_info(parameter, layout_seg, face)
        parameter_gt = get_parameter(depth)
        loss = parameter_loss(parameter, parameter_gt) + \
            discrimitive_loss(parameter, layout_seg, face, average_plane_info, args.delta_v, args.delta_d) + \
            depth_loss(parameter, face, average_plane_info, layout_depth)
        p, q, r, s = parameter 
        depth_mine = get_depth_map(p, q, r, s)
        rms, rel, rlog10, rate_1, rate_2, rate_3 = depth_metrics(depth_mine, layout_depth)
        end = time.time()
        the_time = end - start

        result_string = ('Valid: Epoch: [{} / {}], Batch: [{} / {}], Time: {:.3f}, Loss: {:.4f}\n' \
            + 'rms: {:.3f}, rel: {:.3f}, log10: {:.3f}, delta1: {:.3f}, delta2: {:.3f}, delta3: {:.3f}') \
            .format(epoch + 1, args.epochs, i + 1, len(train_loader), the_time, loss, \
                rms, rel, rlog10, rate_1, rate_2, rate_3)
        write_log(args, epoch, i, 'validation', result_string)
        print(result_string)
 