import numpy as np
import os
from math import *
from PIL import Image
import PIL
import torch
from torchvision import transforms


def get_parameter(device, depth_map, plane_seg):
    '''
    description: get the ground truth parameters(p, q, r, s) from the depth map
    parameter: depth map, plane_seg
    return: the (p, q, r, s) value of all pixels
    '''

    epsilon = 1e-4
    p_list = []
    q_list = []
    r_list = []
    s_list = []
    for i in range(len(depth_map)):
        depth_pivot = depth_map[i][0]
        depth_pivot = torch.clamp(depth_pivot, min = epsilon)
        size_v = len(depth_map[i][0])
        size_u = len(depth_map[i][0][0])
        empty_dv = torch.ones((1, size_u), dtype = float, device = device, requires_grad = False) * epsilon
        empty_du = torch.ones((size_v, 1), dtype = float, device = device, requires_grad = False) * epsilon
        depth_v_up = torch.cat((depth_pivot[1 : , : ], empty_dv), dim = 0)
        depth_v_down = torch.cat((empty_dv, depth_pivot[0 : size_v - 1, : ]), dim = 0)
        depth_u_up = torch.cat((depth_pivot[ : , 1 : ], empty_du), dim = 1)
        depth_u_down = torch.cat((empty_du, depth_pivot[ : , 0 : size_u - 1]), dim = 1)

        frac_pivot = torch.pow(depth_pivot, -1)
        diff_v_up = torch.pow(depth_v_up, -1) - frac_pivot
        diff_v_down = frac_pivot - torch.pow(depth_v_down, -1)
        diff_u_up = torch.pow(depth_u_up, -1) - frac_pivot
        diff_u_down = frac_pivot - torch.pow(depth_u_down, -1)


        false_dv = torch.zeros((1, size_u), dtype = bool, device = device, requires_grad = False)
        false_du = torch.zeros((size_v, 1), dtype = bool, device = device, requires_grad = False)
        mask_v = torch.eq(plane_seg[i][0][1 : , : ], plane_seg[i][0][0 : size_v - 1, :])
        mask_u = torch.eq(plane_seg[i][0][ : , 1 : ], plane_seg[i][0][ : , 0 : size_u - 1])
        mask_v_up = torch.cat((mask_v, false_dv), dim = 0)
        mask_u_up = torch.cat((mask_u, false_du), dim = 1)
        mask_v_down = torch.cat((false_dv, mask_v), dim = 0)
        mask_u_down = torch.cat((false_du, mask_u), dim = 1)
        mask_v_down = mask_v_down & (~ mask_v_up)
        mask_u_down = mask_u_down & (~ mask_u_up)

        #mask_v_down = torch.eq(mask_v_up, 0)
        #mask_u_down = torch.eq(mask_u_up, 0)

        p = diff_u_up * mask_u_up + diff_u_down * mask_u_down
        q = diff_v_up * mask_v_up + diff_v_down * mask_v_down
        v = torch.arange(0, size_v, step = 1, device = device, requires_grad = False).reshape(size_v, 1)
        u = torch.arange(0, size_u, step = 1, device = device, requires_grad = False).reshape(1, size_u)
        r = frac_pivot - p * u - q * v 
        s = torch.sqrt(torch.pow(p, 2) + torch.pow(q, 2) + torch.pow(r, 2))
        p = p / s 
        q = q / s 
        r = r / s

        p = p.unsqueeze(0).unsqueeze(0)
        q = q.unsqueeze(0).unsqueeze(0)
        r = r.unsqueeze(0).unsqueeze(0)
        s = s.unsqueeze(0).unsqueeze(0)
        p_list.append(p)
        q_list.append(q)
        r_list.append(r)
        s_list.append(s)

    p = torch.cat(p_list, dim = 0)
    q = torch.cat(q_list, dim = 0)
    r = torch.cat(r_list, dim = 0)
    s = torch.cat(s_list, dim = 0)
    parameters = torch.cat([p, q, r, s], dim = 1)

    return parameters
    


def get_depth_map(device, parameters):
    '''
    description: get the depth map from the parameters(p, q, r, s)
    parameter: the (p, q, r, s) value of all pixels
    return: evaluated depth map
    '''
    p = parameters[:, 0 : 1]
    q = parameters[:, 1 : 2]
    r = parameters[:, 2 : 3]
    s = parameters[:, 3 : 4]


    size_v = len(p[0][0])
    size_u = len(p[0][0][0])
    vv = torch.arange(0, size_v, step = 1, device = device).reshape(1, 1, size_v, 1)
    uu = torch.arange(0, size_u, step = 1, device = device).reshape(1, 1, 1, size_u)
    depth_map = torch.pow((p * uu + q * vv + r) * s, -1)
    return depth_map

def get_plane_max_num(plane_seg):
    '''
    description: get the plane ids
    parameter: plane seg map
    return: the max num of planes
    '''
    max_num = torch.max(plane_seg)
    return max_num

def get_average_plane_info(device, parameters, plane_seg, max_num):
    '''
    description: get the average plane info 
    parameter: device, parameters per pixel, plane segmentation per pixel, the max segmentation num of planes
    return: average plane info
    '''
    batch_size = len(plane_seg)
    size_v = len(plane_seg[0][0])
    size_u = len(plane_seg[0][0][0])
    average_paramaters = []
    
    for batch in range(batch_size):
        the_parameter = parameters[batch]
        average_paramaters.append([])
        for i in range(max_num + 1):
            the_mask = torch.eq(plane_seg[batch], i) #选择所有seg和i相等的像素
            the_total = torch.sum(the_parameter * the_mask, dim = [1, 2]) #对每个图符合条件的求和
            the_count = torch.sum(the_mask) #求和
            the_count = the_count + torch.eq(the_count, 0) #trick，如果count=0，mask=1，加上变成1(但是total=0，结果还是0)
            average_paramaters[batch].append((the_total / the_count).unsqueeze(0))
        average_paramaters[batch] = torch.cat(average_paramaters[batch], dim = 0).unsqueeze(0)
    average_paramaters = torch.cat(average_paramaters)
    return average_paramaters

def get_average_depth_map(device, plane_seg, average_plane_info):
    '''
    description: get the depth from the average parameters(p, q, r, s)
    parameter: the plane ids, plane_segs, the average plane infos
    return: evaluated depth maps of all planes
    '''
    depth_map = []
    batch_size = len(plane_seg)
    size_v = len(plane_seg[0][0])
    size_u = len(plane_seg[0][0][0])

    for i in range(batch_size):
        depth_map.append([])
        for the_id in range(len(average_plane_info[i])):
            the_id = int(the_id) 
            p = average_plane_info[i][the_id][0]
            q = average_plane_info[i][the_id][1]
            r = average_plane_info[i][the_id][2]
            s = average_plane_info[i][the_id][3]
            v = torch.arange(0, size_v, step = 1, device = device).reshape(size_v, 1).repeat(1, size_u)
            u = torch.arange(0, size_u, step = 1, device = device).reshape(1, size_u).repeat(size_v, 1)

            mask = torch.eq(plane_seg[i][0], the_id)
            reverse_mask = torch.ne(plane_seg[i][0], the_id)
            raw_result = (u * p + v * q + r) * s 

            raw_result = raw_result + reverse_mask #trick, 防止/0
            raw_result = torch.pow(raw_result, -1)
            depth_map[i].append((mask * raw_result).unsqueeze(0))


        depth_map[i] = torch.cat(depth_map[i])
        depth_map[i] = torch.sum(depth_map[i], dim = 0, keepdim = True).unsqueeze(0)
    depth_map = torch.cat(depth_map)
    return depth_map

def set_average_plane_info(device, plane_seg, average_plane_info):
    '''
    description: set the per pixel plane info to the average
    parameter: the plane ids, plane_segs, the average plane infos, the shape of the depth map
    return: average per pixel plane info
    '''
    batch_size = len(plane_seg)
    size_v = len(plane_seg[0][0])
    size_u = len(plane_seg[0][0][0])
    new_paramater = []
    for i in range(batch_size):
        new_paramater.append([])
        for the_id in range(len(average_plane_info[i])):
            the_id = int(the_id) 
            mask = torch.eq(plane_seg[i][0], the_id)
            p = average_plane_info[i][the_id][0]
            q = average_plane_info[i][the_id][1]
            r = average_plane_info[i][the_id][2]
            s = average_plane_info[i][the_id][3]
            masked_p = (mask * p).unsqueeze(0)
            masked_q = (mask * q).unsqueeze(0)
            masked_r = (mask * r).unsqueeze(0)
            masked_s = (mask * s).unsqueeze(0)
            the_parameter = torch.cat([masked_p, masked_q, masked_r, masked_s]).unsqueeze(0)
            new_paramater[i].append(the_parameter)

        new_paramater[i] = torch.cat(new_paramater[i])
        new_paramater[i] = torch.sum(new_paramater[i], dim = 0, keepdim = True)
    new_paramater = torch.cat(new_paramater)
    return new_paramater

def get_plane_ids(plane_seg):
    '''
    description: get the plane ids
    parameter: plane seg map
    return: plane ids
    '''
    batch_size = len(plane_seg)
    plane_ids = []
    for i in range(batch_size):
        plane_ids.append([])
        for v in range(len(plane_seg[i][0])):
            for u in range(len(plane_seg[i][0][v])): 
                the_seg = int(plane_seg[i][0][v][u])
                if not the_seg in plane_ids[i]:
                    plane_ids[i].append(the_seg)
    return plane_ids



#unit test code
def utils_test():
    transform_depth = transforms.Compose([transforms.Resize([152, 114]), transforms.ToTensor()])
    transform_original = transforms.Compose([transforms.ToTensor()])
    transform_seg = transforms.Compose([transforms.Resize([152, 114], interpolation = PIL.Image.NEAREST), transforms.ToTensor()])
    name = 'E:\\dataset\\geolayout\\training\\layout_depth\\0b2156c0034b43bc8b06023a4c4fe2db_i1_2_layout.png'
    depth_map_original_0_load = Image.open(name).convert('I')
    depth_map_original_0 = transform_depth(depth_map_original_0_load) / 4000.0
    name = 'E:\\dataset\\geolayout\\training\\layout_depth\\0b124e1ec3bf4e6fb2ec42f179cc9ff0_i1_5_layout.png' 
    depth_map_original_1_load = Image.open(name).convert('I')
    depth_map_original_1 = transform_depth(depth_map_original_1_load) / 4000.0
    depth_map_original = torch.stack((depth_map_original_0, depth_map_original_1))
    name = 'E:\\dataset\\geolayout\\training\\layout_seg\\0b2156c0034b43bc8b06023a4c4fe2db_i1_2_seg.png'
    plane_seg_0_load = Image.open(name).convert('I')
    plane_seg_0 = transform_seg(plane_seg_0_load)
    name = 'E:\\dataset\\geolayout\\training\\layout_seg\\0b124e1ec3bf4e6fb2ec42f179cc9ff0_i1_5_seg.png'
    plane_seg_1_load = Image.open(name).convert('I')
    plane_seg_1 = transform_seg(plane_seg_1_load)
    plane_seg = torch.stack((plane_seg_0, plane_seg_1)) 
    plane_ids = get_plane_ids(plane_seg) 
    device = torch.device("cpu")

    parameters = get_parameter(device, depth_map_original, plane_seg)


    depth_map = get_depth_map(device, parameters)
    print(parameters.shape) 
    print(depth_map_original.shape, depth_map.shape)
    rate = 0
    for i in range(len(depth_map)):
        for v in range(len(depth_map[i][0])):
            for u in range(len(depth_map[i][0][v])): 
                diff = abs(float(depth_map[i][0][v][u]) - float(depth_map_original[i][0][v][u]))
                if diff < 0.5:  
                    rate += 1 
    rate /= (len(depth_map) * len(depth_map[0][0]) * len(depth_map[0][0][0]))
    print(rate)


    max_num = get_plane_max_num(plane_seg)
    plane_info = get_average_plane_info(device, parameters, plane_seg, max_num)
    depth_average = get_average_depth_map(device, plane_seg, plane_info)

    for i in range(len(plane_ids)):
        avg = torch.mean(depth_map_original[i])
        for the_id in plane_ids[i]:
            count = 0
            diff = 0
            for v in range(len(depth_map[i][0])):
                for u in range(len(depth_map[i][0][v])): 
                    
                    if int(plane_seg[i][0][v][u]) == the_id:
                        count += 1
                        diff += abs(float(depth_average[i][0][v][u]) - float(depth_map_original[i][0][v][u]))
            diff /= count 
            diff /= avg
            print(i, the_id, count, diff)

    parameter_average = set_average_plane_info(device, plane_seg, plane_info)
    for i in range(len(plane_ids)):
        avg_p = torch.mean(parameters[i][0])
        avg_q = torch.mean(parameters[i][1])
        avg_r = torch.mean(parameters[i][2])
        avg_s = torch.mean(parameters[i][3])

        for the_id in plane_ids[i]:
            count = 0
            dp = 0
            dq = 0
            dr = 0
            ds = 0
            for v in range(len(depth_map[i][0])):
                for u in range(len(depth_map[i][0][v])): 
                    if int(plane_seg[i][0][v][u]) == the_id:
                        count += 1
                        dp += abs(float(parameter_average[i][0][v][u]) - float(parameters[i][0][v][u]))
                        dq += abs(float(parameter_average[i][1][v][u]) - float(parameters[i][1][v][u]))
                        dr += abs(float(parameter_average[i][2][v][u]) - float(parameters[i][2][v][u]))
                        ds += abs(float(parameter_average[i][3][v][u]) - float(parameters[i][3][v][u]))

            dp /= count 
            dq /= count
            dr /= count
            ds /= count
            dp /= avg_p
            dq /= avg_q
            dr /= avg_r 
            ds /= avg_s
            print(i, the_id, count, dp, dq, dr, ds)

#utils_test()