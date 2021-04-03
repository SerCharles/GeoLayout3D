import numpy as np
import os
from math import *
from PIL import Image
import PIL
import torch
from torchvision import transforms
from utils.get_parameter_geolayout import *

def parameter_loss(parameter, parameter_gt):
    '''
    description: get the parameter loss 
    parameter: the parameter driven by our model, the ground truth parameter
    return: parameter loss
    '''
    loss = torch.sum(torch.abs(parameter - parameter_gt))
    loss /= (len(parameter) * len(parameter[0][0]) * len(parameter[0][0][0]))
    return loss

def discrimitive_loss(parameters, plane_seg_gt, average_plane_info, delta_v, delta_d):
    '''
    description: get the discrimitive loss 
    parameter: the parameter driven by our model, the ground truth segmentation, the ground truth plane id
        the average plane info, threshold of the same plane, threshold of different planes
    return: depth loss
    '''
    batch_size = len(parameters)
    lvar = [] 
    dvar = []

    for i in range(batch_size):
        p = parameters[i][0]
        q = parameters[i][1]
        r = parameters[i][2]
        s = parameters[i][3]
        useful_mask = []

        current_lvar = []
        for seg_id in range(len(average_plane_info[i])):
            mask = torch.eq(plane_seg_gt[i][0], seg_id) 
            count = mask.sum()
            useful_mask.append(torch.ne(count, 0).unsqueeze(0))
            dp_total = mask * torch.abs(p - average_plane_info[i][seg_id][0])
            dq_total = mask * torch.abs(q - average_plane_info[i][seg_id][1])
            dr_total = mask * torch.abs(r - average_plane_info[i][seg_id][2])
            ds_total = mask * torch.abs(s - average_plane_info[i][seg_id][3])
            loss_total = torch.clamp(dp_total + dq_total + dr_total + ds_total - delta_v, min = 0)
            mask_auxiliary = torch.eq(count, 0) #trick
            count = count + mask_auxiliary
            the_sum = loss_total.sum() / count 
            current_lvar.append(the_sum.unsqueeze(0))
        useful_mask = torch.cat(useful_mask)
        current_lvar = torch.cat(current_lvar)
        C = useful_mask.sum()
        lvar.append((current_lvar.sum() / C).unsqueeze(0))

        current_dvar = []
        for ii in range(len(average_plane_info[i]) - 1):
            for jj in range(ii + 1, len(average_plane_info[i])):
                diff_param = torch.abs(average_plane_info[i][ii] - average_plane_info[i][jj])
                diff = diff_param.sum()
                the_sum = torch.clamp(delta_d - diff, min = 0)
                the_sum = the_sum * useful_mask[ii] * useful_mask[jj]
                current_dvar.append(the_sum.unsqueeze(0))
        current_dvar = torch.cat(current_dvar) 
        dvar_result = current_dvar.sum() * 2 / C / (C - 1)
        dvar.append(dvar_result.unsqueeze(0))
    lvar = torch.cat(lvar)
    dvar = torch.cat(dvar)
    total_loss = (lvar + dvar).sum() / batch_size

    return total_loss
  

def depth_loss(depth, depth_gt):
    '''
    description: get the depth loss 
    parameter: the depth calculated by the average plane info, ground truth
    return: depth loss
    '''
    loss = torch.sum(torch.abs(depth - depth_gt))
    loss /= (len(depth) * len(depth[0][0]) * len(depth[0][0][0]))
    return loss





#unit test code
def loss_test():
    transform_depth = transforms.Compose([transforms.Resize([152, 114]), transforms.ToTensor()])
    transform_seg = transforms.Compose([transforms.Resize([152, 114], interpolation = PIL.Image.NEAREST), transforms.ToTensor()])
    name = 'E:\\dataset\\geolayout\\training\\layout_depth\\0b2156c0034b43bc8b06023a4c4fe2db_i1_2_layout.png'
    depth_map_original_0 = Image.open(name).convert('I')
    depth_map_original_0 = transform_depth(depth_map_original_0) / 4000.0
    name = 'E:\\dataset\\geolayout\\training\\layout_depth\\0b124e1ec3bf4e6fb2ec42f179cc9ff0_i1_5_layout.png' 
    depth_map_original_1 = Image.open(name).convert('I')
    depth_map_original_1 = transform_depth(depth_map_original_1) / 4000.0
    depth_map_original = torch.stack((depth_map_original_0, depth_map_original_1))
    name = 'E:\\dataset\\geolayout\\training\\layout_seg\\0b2156c0034b43bc8b06023a4c4fe2db_i1_2_seg.png'
    plane_seg_0 = Image.open(name).convert('I')
    plane_seg_0 = transform_seg(plane_seg_0)
    name = 'E:\\dataset\\geolayout\\training\\layout_seg\\0b124e1ec3bf4e6fb2ec42f179cc9ff0_i1_5_seg.png'
    plane_seg_1 = Image.open(name).convert('I')
    plane_seg_1 = transform_seg(plane_seg_1)
    plane_seg = torch.stack((plane_seg_0, plane_seg_1)) 
    plane_ids = get_plane_ids(plane_seg) 


    device = torch.device("cpu")
    parameters = get_parameter(device, depth_map_original, plane_seg)
    depth_map = get_depth_map(device, parameters)
    max_num = get_plane_max_num(plane_seg)
    plane_info = get_average_plane_info(device, parameters, plane_seg, max_num)
    depth_average = get_average_depth_map(device, plane_seg, plane_info)
    parameters_avg = set_average_plane_info(device, plane_seg, plane_info)

    loss_p = parameter_loss(parameters_avg, parameters)
    loss_dis_1 = discrimitive_loss(parameters, plane_seg, plane_info, 0.1, 1.0)
    loss_dis_2 = discrimitive_loss(parameters_avg, plane_seg, plane_info, 0.1, 1.0)
    loss_d = depth_loss(depth_average, depth_map)
    print(loss_p, loss_dis_1, loss_dis_2, loss_d)
#loss_test()