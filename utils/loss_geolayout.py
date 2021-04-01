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
    loss /= (len(parameter[0]) * len(parameter[0][0]) * len(parameter[0][0][0]))
    return loss

def discrimitive_loss(parameters, plane_seg_gt, plane_id_gt, average_plane_info, delta_v, delta_d):
    '''
    description: get the discrimitive loss 
    parameter: the parameter driven by our model, the ground truth segmentation, the ground truth plane id
        the average plane info, threshold of the same plane, threshold of different planes
    return: depth loss
    '''
    batch_size = len(parameters)
    total_loss = 0
    for i in range(batch_size):
        C = len(plane_id_gt[i])
        lvars = {}
        lvar = 0
        dvar = 0
        for j in range(C):
            t_id = int(plane_id_gt[i][j])
            lvars[t_id] = {'count': 0, 'loss': 0.0} 
    
        p = parameters[i][0]
        q = parameters[i][1]
        r = parameters[i][2]
        s = parameters[i][3]
        #get lvars
        for v in range(len(plane_seg_gt[i][0])) :
            for u in range(len(plane_seg_gt[i][0][v])): 
                the_seg = int(plane_seg_gt[i][0][v][u])
                the_p = float(p[v][u])
                the_q = float(q[v][u])
                the_r = float(r[v][u])
                the_s = float(s[v][u])
                gt_p = average_plane_info[i][the_seg]['p']
                gt_q = average_plane_info[i][the_seg]['q']
                gt_r = average_plane_info[i][the_seg]['r']
                gt_s = average_plane_info[i][the_seg]['s']

                dp = abs(the_p - gt_p)
                dq = abs(the_q - gt_q)
                dr = abs(the_r - gt_r)
                ds = abs(the_s - gt_s)

                loss = max(0, dp + dq + dr + ds - delta_v)
                lvars[the_seg]['count'] += 1 
                lvars[the_seg]['loss'] += loss 
        for the_id in plane_id_gt[i]: 
            the_id = int(the_id)
            the_average = lvars[the_id]['loss'] / lvars[the_id]['count']
            lvar += the_average
        lvar /= C 

        #get dvar 
        for ii in range(C - 1):
            for jj in range(i + 1, C):
                id_i = plane_id_gt[i][ii]
                id_j = plane_id_gt[i][jj]
                pi = average_plane_info[i][id_i]['p']
                qi = average_plane_info[i][id_i]['q']
                ri = average_plane_info[i][id_i]['r']
                si = average_plane_info[i][id_i]['s']
                pj = average_plane_info[i][id_j]['p']
                qj = average_plane_info[i][id_j]['q']
                rj = average_plane_info[i][id_j]['r']
                sj = average_plane_info[i][id_j]['s']

                dp = abs(pi - pj)
                dq = abs(qi - qj) 
                dr = abs(ri - rj)
                ds = abs(si - sj)

                loss = max(0, delta_d - dp - dq - dr - ds)
                dvar += loss 
        dvar /= (C * (C - 1))
        total_loss += lvar 
        total_loss += dvar
    total_loss /= batch_size


    batch_size = len(parameters)
    lvar = torch.Tensor(0.0, requires_grad = True)
    dvar = torch.Tensor(0.0, requires_grad = True)

    for i in range(batch_size):
        p = parameters[i][0]
        q = parameters[i][1]
        r = parameters[i][2]
        s = parameters[i][3]
        useful_mask = torch.zeros((len(average_plane_info[i])))

        current_lvar = torch.Tensor(0.0, requires_grad = True)
        for seg_id in range(len(average_plane_info[i])):
            mask = torch.eq(plane_seg_gt[i][0], seg_id) 
            count = mask.sum()
            useful_mask[seg_id] = torch.ne(count, 0)
            dp_total = mask * torch.abs(p - average_plane_info[i][seg_id][0])
            dq_total = mask * torch.abs(q - average_plane_info[i][seg_id][1])
            dr_total = mask * torch.abs(r - average_plane_info[i][seg_id][2])
            ds_total = mask * torch.abs(s - average_plane_info[i][seg_id][3])
            loss_total = torch.clamp(dp_total + dq_total + dr_total + ds_total - delta_v, min = 0)
            mask_auxiliary = torch.eq(count, 0) #trick
            count = count + mask_auxiliary
            the_sum = loss_total.sum() / count 
            current_lvar += the_sum 
        C = useful_mask.sum()
        current_lvar = current_lvar / C

        current_dvar = torch.Tensor(0.0, requires_grad = True)
        for ii in range(len(average_plane_info[i])):
            for jj in range(len(average_plane_info[i])):
                diff_param = average_plane_info[i][ii] - average_plane_info[i][jj]
                the_sum = torch.clamp(delta_d - diff_param.sum(), min = 0)
                current_dvar += the_sum 
        current_dvar = current_dvar / (C * (C - 1))
        lvar += current_lvar
        dvar += current_dvar
    total_loss = (lvar + dvar) / batch_size


    return total_loss
  

def depth_loss(depth, depth_gt):
    '''
    description: get the depth loss 
    parameter: the depth calculated by the average plane info, ground truth
    return: depth loss
    '''
    loss = torch.sum(torch.abs(depth - depth_gt))
    loss /= (len(depth[0]) * len(depth[0][0]) * len(depth[0][0][0]))
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

    plane_ids = get_plane_ids(plane_seg)


    parameters = get_parameter(depth_map_original, plane_seg)
    depth_map = get_depth_map(parameters)
    plane_info = get_average_plane_info(parameters, plane_seg, plane_ids)
    depth_average = get_average_depth_map(plane_ids, plane_seg, plane_info)
    parameters_avg = set_average_plane_info(plane_ids, plane_seg, plane_info)

    loss_p = parameter_loss(parameters_avg, parameters)
    loss_dis_1 = discrimitive_loss(parameters, plane_seg, plane_ids, plane_info, 0.1, 1.0)
    loss_dis_2 = discrimitive_loss(parameters_avg, plane_seg, plane_ids, plane_info, 0.1, 1.0)
    loss_d = depth_loss(plane_ids, plane_seg, plane_info, depth_map)
    print(loss_p, loss_dis_1, loss_dis_2, loss_d)
#loss_test()