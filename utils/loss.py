''' 
The implementation of parameter loss, discrimitive loss and depth loss
'''

import numpy as np
import os
from math import *
from PIL import Image
import PIL
import torch
from torchvision import transforms
from utils.utils import *

def parameter_loss(parameter, parameter_gt):
    '''
    description: get the parameter loss 
    parameter: the parameter driven by our model, the ground truth parameter
    return: parameter loss
    '''
    loss = torch.sum(torch.abs(parameter - parameter_gt)) / ((len(parameter) * len(parameter[0][0]) * len(parameter[0][0][0])))
    return loss

def discrimitive_loss(parameters, plane_seg_gt, average_plane_info, delta_v, delta_d):
    '''
    description: get the discrimitive loss 
    parameter: the parameter driven by our model, the ground truth segmentation, the ground truth plane id
        the average plane info, threshold of the same plane, threshold of different planes
    return: depth loss
    '''
    batch_size = len(plane_seg_gt)
    max_num = len(average_plane_info[0])
    lvar = [] 
    dvar = []

    for i in range(batch_size):
        p = parameters[i][0]
        q = parameters[i][1]
        r = parameters[i][2]
        s = parameters[i][3]
        useful_mask = []

        current_lvar = []
        for seg_id in range(max_num):
            mask = torch.eq(plane_seg_gt[i][0], seg_id) 
            mask = mask.detach()

            count = mask.sum()
            useful_mask.append(torch.ne(count, 0).unsqueeze(0))
            dp_total = mask * torch.abs(p - average_plane_info[i][seg_id][0])
            dq_total = mask * torch.abs(q - average_plane_info[i][seg_id][1])
            dr_total = mask * torch.abs(r - average_plane_info[i][seg_id][2])
            ds_total = mask * torch.abs(s - average_plane_info[i][seg_id][3])
            loss_total = torch.clamp(dp_total + dq_total + dr_total + ds_total - delta_v, min = 0)
            mask_auxiliary = torch.eq(count, 0) #trick
            new_count = count + mask_auxiliary
            new_count = new_count.detach()


            the_sum = loss_total.sum() / new_count 
            current_lvar.append(the_sum.unsqueeze(0))
        useful_mask = torch.cat(useful_mask)
        useful_mask = useful_mask.detach()
        current_lvar = torch.cat(current_lvar)
        C = useful_mask.sum()
        C = C.detach()
        
        lvar.append((current_lvar.sum() / C).unsqueeze(0))


        current_dvar = []
        for ii in range(max_num - 1):
            for jj in range(ii + 1, max_num):
                diff_param = torch.abs(average_plane_info[i][ii] - average_plane_info[i][jj])
                diff = diff_param.sum()
                the_sum = torch.clamp(delta_d - diff, min = 0)
                masked_sum = the_sum * useful_mask[ii] * useful_mask[jj]
                current_dvar.append(masked_sum.unsqueeze(0))
        current_dvar = torch.cat(current_dvar) 
        
        dvar_raw = current_dvar.sum() * 2
        #dvar_raw = current_dvar.sum()
        #防止/1
        divided = C * (C - 1)
        divided_mask = torch.eq(divided, 0)
        divided_total = divided + divided_mask 
        divided_total = divided_total.detach()
        
        dvar_result = dvar_raw / divided_total
        dvar.append(dvar_result.unsqueeze(0))

    lvar = torch.cat(lvar)
    dvar = torch.cat(dvar)



    total_loss = (lvar + dvar).sum() / batch_size

    return total_loss
  

def depth_loss(depth, depth_gt, epsilon):
    '''
    description: get the depth loss 
    parameter: the depth calculated by the average plane info, ground truth
    return: depth loss
    '''
    small_enough = 1e-4
    small_mask = torch.abs(depth) < small_enough
    not_small_mask = ~small_mask
    depth_not_small = depth * not_small_mask + small_mask #0的点值变成1，其余不变
    depth_frac_not_small = torch.pow(depth_not_small, -1) * not_small_mask #0的点变成0， 其余正常取倒数

    depth_small = depth * small_mask 
    depth_frac_small = depth_small * (-1 / small_enough / small_enough)
    depth_frac = depth_frac_not_small + depth_frac_small

    depth_gt_frac = torch.pow(depth_gt + epsilon, -1)
    loss = torch.sum(torch.abs(depth_frac - depth_gt_frac)) / ((len(depth) * len(depth[0][0]) * len(depth[0][0][0])))
    return loss



def depth_loss_direct(device, plane_seg, average_plane_info, depth_gt, epsilon):
    '''
    description: get the depth loss directly
    parameter: the depth calculated by the average plane info, ground truth
    return: depth loss
    '''
    depth_map_frac = []
    batch_size = len(plane_seg)
    size_v = len(plane_seg[0][0])
    size_u = len(plane_seg[0][0][0])

    for i in range(batch_size):
        depth_map_frac.append([])
        for the_id in range(len(average_plane_info[i])):
            the_id = int(the_id) 
            p = average_plane_info[i][the_id][0]
            q = average_plane_info[i][the_id][1]
            r = average_plane_info[i][the_id][2]
            s = average_plane_info[i][the_id][3]
            v = torch.arange(0, size_v, step = 1, requires_grad = False).reshape(size_v, 1)
            u = torch.arange(0, size_u, step = 1, requires_grad = False).reshape(1, size_u)
            if device: 
                v = v.cuda() 
                u = u.cuda()

            mask = torch.eq(plane_seg[i][0], the_id)
            mask = mask.detach()

            raw_result = (u * p + v * q + r) * s 
            result = raw_result * mask
            depth_map_frac[i].append(result.unsqueeze(0))


        depth_map_frac[i] = torch.cat(depth_map_frac[i])
        depth_map_frac[i] = torch.sum(depth_map_frac[i], dim = 0, keepdim = True).unsqueeze(0)
    depth_map_frac = torch.cat(depth_map_frac)

    depth_gt_frac = torch.pow(depth_gt + epsilon, -1)
    loss = torch.sum(torch.abs(depth_map_frac - depth_gt_frac)) / ((len(depth_gt) * len(depth_gt[0][0]) * len(depth_gt[0][0][0])))
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


    device = False
    parameters = get_parameter(device, depth_map_original, plane_seg, 1e-8)
    depth_map = get_depth_map(device, parameters, 1e-8)
    max_num = get_plane_max_num(plane_seg)
    plane_info = get_average_plane_info(device, parameters, plane_seg, max_num)
    depth_average = get_average_depth_map(device, plane_seg, plane_info, 1e-8)
    parameters_avg = set_average_plane_info(plane_seg, plane_info)

    loss_p = parameter_loss(parameters_avg, parameters)
    loss_dis_1 = discrimitive_loss(parameters, plane_seg, plane_info, 0.1, 1.0)
    loss_dis_2 = discrimitive_loss(parameters_avg, plane_seg, plane_info, 0.1, 1.0)
    loss_d = depth_loss(depth_average, depth_map, 1e-8)
    loss_d_mod = depth_loss_direct(device, plane_seg, plane_info, depth_map, 1e-8)
    print(loss_p, loss_dis_1, loss_dis_2, loss_d, loss_d_mod)
#loss_test()