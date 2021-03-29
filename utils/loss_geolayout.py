import numpy as np
import os
from math import *
from PIL import Image
import PIL
import torch
from torchvision import transforms
from get_parameter_geolayout import *

def parameter_loss(parameter, parameter_gt):
    '''
    description: get the parameter loss 
    parameter: the parameter driven by our model, the ground truth parameter
    return: parameter loss
    '''
    p, q, r, s = parameter
    p_gt, q_gt, r_gt, s_gt = parameter_gt
    dp = torch.sum((torch.abs(p - p_gt)))
    dq = torch.sum((torch.abs(q - q_gt)))
    dr = torch.sum((torch.abs(r - r_gt))) 
    ds = torch.sum((torch.abs(s - s_gt)))
    loss = dp + dq + dr + ds 
    loss /= (len(p[0]) * len(p[0][0]))
    return float(loss) 

def discrimitive_loss(parameters, plane_seg_gt, plane_id_gt, average_plane_info, delta_v, delta_d):
    '''
    description: get the discrimitive loss 
    parameter: the parameter driven by our model, the ground truth segmentation loss, the ground truth plane id
        the average plane info, threshold of the same plane, threshold of different planes
    return: depth loss
    '''
    p, q, r, s = parameters
    C = len(plane_id_gt)
    lvars = {}
    lvar = 0
    dvar = 0
    for i in range(C):
        t_id = plane_id_gt[i]
        lvars[t_id] = {'count': 0, 'loss': 0.0} 
    
    #get lvars
    for v in range(len(plane_seg[0])) :
        for u in range(len(plane_seg[0][v])): 
            the_seg = int(plane_seg[0][v][u])
            the_p = float(p[0][v][u])
            the_q = float(q[0][v][u])
            the_r = float(r[0][v][u])
            the_s = float(s[0][v][u])
            gt_p = average_plane_info[the_seg]['p']
            gt_q = average_plane_info[the_seg]['q']
            gt_r = average_plane_info[the_seg]['r']
            gt_s = average_plane_info[the_seg]['s']

            dp = abs(the_p - gt_p)
            dq = abs(the_q - gt_q)
            dr = abs(the_r - gt_r)
            ds = abs(the_s - gt_s)

            loss = max(0, dp + dq + dr + ds - delta_v)
            lvars[the_seg]['count'] += 1 
            lvars[the_seg]['loss'] += loss 
    for the_id in plane_id_gt: 
        the_average = lvars[the_id]['loss'] / lvars[the_id]['count']
        lvar += the_average
    lvar /= C 

    #get dvar 
    for i in range(C - 1):
        for j in range(i + 1, C):
            id_i = plane_id_gt[i]
            id_j = plane_id_gt[j]
            pi = average_plane_info[id_i]['p']
            qi = average_plane_info[id_i]['q']
            ri = average_plane_info[id_i]['r']
            si = average_plane_info[id_i]['s']
            pj = average_plane_info[id_j]['p']
            qj = average_plane_info[id_j]['q']
            rj = average_plane_info[id_j]['r']
            sj = average_plane_info[id_j]['s']

            dp = abs(pi - pj)
            dq = abs(qi - qj) 
            dr = abs(ri - rj)
            ds = abs(si - sj)

            loss = max(0, delta_d - dp - dq - dr - ds)
            dvar += loss 
    dvar /= (C * (C - 1))
    return lvar + dvar
  

def depth_loss(plane_id_gt, plane_seg_gt, average_plane_info, depth_gt):
    '''
    description: get the depth loss 
    parameter: the ground truth plane ids and seg infos, the average pqrs info, ground truth
    return: depth loss
    '''
    depth = get_average_depth_map(plane_id_gt, plane_seg_gt, average_plane_info)

    loss = torch.sum(torch.abs(depth - depth_gt))
    loss /= (len(depth[0]) * len(depth[0][0]))
    return float(loss)

'''
name = 'E:\\dataset\\geolayout\\validation\\layout_depth\\04cdd02138664b138f281bb5ad8b957f_i1_3_layout.png'
depth_map_original = Image.open(name).convert('I')
transform_depth = transforms.Compose([transforms.Resize([152, 114]), transforms.ToTensor()])
depth_map_original = transform_depth(depth_map_original) / 4000.0
name = 'E:\\dataset\\geolayout\\validation\\layout_seg\\04cdd02138664b138f281bb5ad8b957f_i1_3_seg.png'
plane_seg = Image.open(name).convert('I')
transform_seg = transforms.Compose([transforms.Resize([152, 114], interpolation = PIL.Image.NEAREST), transforms.ToTensor()])
plane_seg = transform_seg(plane_seg)
plane_ids = [1, 3, 4, 5]


p, q, r, s = get_parameter(depth_map_original, plane_seg)
depth_map = get_depth_map(p, q, r, s)
plane_info = get_average_plane_info((p, q, r, s), plane_seg, plane_ids)
depth_average = get_average_depth_map(plane_ids, plane_seg, plane_info)
p_avg, q_avg, r_avg, s_avg = set_average_plane_info(plane_ids, plane_seg, plane_info)

loss_p = parameter_loss((p_avg, q_avg, r_avg, s_avg), (p, q, r, s))
loss_dis_1 = discrimitive_loss((p, q, r, s), plane_seg, plane_ids, plane_info, 0.1, 1.0)
loss_dis_2 = discrimitive_loss((p_avg, q_avg, r_avg, s_avg), plane_seg, plane_ids, plane_info, 0.1, 1.0)
loss_d = depth_loss(plane_ids, plane_seg, plane_info, depth_map)
print(loss_p, loss_dis_1, loss_dis_2, loss_d)
'''