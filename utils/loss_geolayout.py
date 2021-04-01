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
    return total_loss
  

def depth_loss(plane_id_gt, plane_seg_gt, average_plane_info, depth_gt):
    '''
    description: get the depth loss 
    parameter: the ground truth plane ids and seg infos, the average pqrs info, ground truth
    return: depth loss
    '''
    depth = get_average_depth_map(plane_id_gt, plane_seg_gt, average_plane_info)
    loss = torch.sum(torch.abs(depth - depth_gt))
    loss /= (len(depth[0]) * len(depth[0][0]) * len(depth[0][0][0]))
    return loss





#unit test code
def loss_test():
    transform_depth = transforms.Compose([transforms.Resize([152, 114]), transforms.ToTensor()])
    transform_seg = transforms.Compose([transforms.Resize([152, 114], interpolation = PIL.Image.NEAREST), transforms.ToTensor()])
    name = 'E:\\dataset\\geolayout\\validation\\layout_depth\\04cdd02138664b138f281bb5ad8b957f_i1_3_layout.png'
    depth_map_original_0 = Image.open(name).convert('I')
    depth_map_original_0 = transform_depth(depth_map_original_0) / 4000.0
    name = 'E:\\dataset\\geolayout\\validation\\layout_depth\\075307518bc2495498609ee2ff6dd003_i1_2_layout.png'
    depth_map_original_1 = Image.open(name).convert('I')
    depth_map_original_1 = transform_depth(depth_map_original_1) / 4000.0
    depth_map_original = torch.stack((depth_map_original_0, depth_map_original_1))
    name = 'E:\\dataset\\geolayout\\validation\\layout_seg\\04cdd02138664b138f281bb5ad8b957f_i1_3_seg.png'
    plane_seg_0 = Image.open(name).convert('I')
    plane_seg_0 = transform_seg(plane_seg_0)
    name = 'E:\\dataset\\geolayout\\validation\\layout_seg\\075307518bc2495498609ee2ff6dd003_i1_2_seg.png'
    plane_seg_1 = Image.open(name).convert('I')
    plane_seg_1 = transform_depth(plane_seg_1)
    plane_seg = torch.stack((plane_seg_0, plane_seg_1))  
    print(depth_map_original.size(), plane_seg.size())

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