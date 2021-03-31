import numpy as np
import os
from math import *
from PIL import Image
import PIL
import torch
from torchvision import transforms


def get_parameter(depth_map, plane_seg):
    '''
    description: get the ground truth parameters(p, q, r, s) from the depth map
    parameter: depth map, plane_seg
    return: the (p, q, r, s) value of all pixels
    '''
    p = torch.zeros(depth_map.size(), dtype = float)  
    q = torch.zeros(depth_map.size(), dtype = float)  
    r = torch.zeros(depth_map.size(), dtype = float)  
    s = torch.zeros(depth_map.size(), dtype = float)  
    for i in range(len(depth_map)):
        for v in range(len(depth_map[i][0])):
            for u in range(len(depth_map[i][0][v])): 
                zz = float(depth_map[i][0][v][u]) 

                if u != len(depth_map[i][0][v]) - 1 and int(plane_seg[i][0][v][u + 1]) == int(plane_seg[i][0][v][u]):
                    pp = 1.0 / float(depth_map[i][0][v][u + 1]) - 1.0 / float(depth_map[i][0][v][u])
                else: 
                    pp = 1.0 / float(depth_map[i][0][v][u]) - 1.0 / float(depth_map[i][0][v][u - 1])
                if v != len(depth_map[i][0]) - 1 and int(plane_seg[i][0][v + 1][u]) == int(plane_seg[i][0][v][u]):
                    qq = 1.0 / float(depth_map[i][0][v + 1][u]) - 1.0 / float(depth_map[i][0][v][u])
                else: 
                    qq = 1.0 / float(depth_map[i][0][v][u]) - 1.0 / float(depth_map[i][0][v - 1][u])
                rr = 1 / zz - pp * u - qq * v
                ss = sqrt(pp ** 2 + qq ** 2 + rr ** 2) 
                pp /= ss 
                qq /= ss 
                rr /= ss 
                p[i][0][v][u] = pp 
                q[i][0][v][u] = qq 
                r[i][0][v][u] = rr 
                s[i][0][v][u] = ss 
    parameters = torch.cat((p, q, r, s), dim = 1)
    return parameters

def get_depth_map(parameters):
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
    depth_map = torch.zeros(p.size(), dtype = float) 
    vv = torch.Tensor(range(size_v)).reshape(1, 1, size_v, 1)
    uu = torch.Tensor(range(size_u)).reshape(1, 1, 1, size_u)
    depth_map = torch.pow((p * uu + q * vv + r) * s, -1)
    return depth_map

def get_average_plane_info(parameters, plane_seg, plane_ids):
    '''
    description: get the average plane info 
    parameter: parameters per pixel, plane segmentation per pixel, the list of plane ids 
    return: average plane info
    '''
    batch_size = len(plane_seg)
    average_paramaters = []
    for i in range(batch_size):
        average_paramaters.append({})
        for t_id in plane_ids[i]: 
            t_id = int(t_id) 
            average_paramaters[i][t_id] = {'count': 0, 'p': 0.0, 'q': 0.0, 'r': 0.0, 's': 0.0}
        p = parameters[i][0]
        q = parameters[i][1]
        r = parameters[i][2]
        s = parameters[i][3]

        for v in range(len(plane_seg[i][0])) :
            for u in range(len(plane_seg[i][0][v])): 
                the_seg = int(plane_seg[i][0][v][u]) 
                the_p = float(p[v][u])
                the_q = float(q[v][u])
                the_r = float(r[v][u])
                the_s = float(s[v][u])
                average_paramaters[i][the_seg]['count'] += 1
                average_paramaters[i][the_seg]['p'] += the_p
                average_paramaters[i][the_seg]['q'] += the_q
                average_paramaters[i][the_seg]['r'] += the_r
                average_paramaters[i][the_seg]['s'] += the_s
            
        for the_id in plane_ids[i]: 
            the_id = int(the_id)
            pp = average_paramaters[i][the_id]['p'] / average_paramaters[i][the_id]['count']
            qq = average_paramaters[i][the_id]['q'] / average_paramaters[i][the_id]['count']
            rr = average_paramaters[i][the_id]['r'] / average_paramaters[i][the_id]['count']      
            ss = average_paramaters[i][the_id]['s'] / average_paramaters[i][the_id]['count']

            average_paramaters[i][the_id]['p'] = pp
            average_paramaters[i][the_id]['q'] = qq
            average_paramaters[i][the_id]['r'] = rr
            average_paramaters[i][the_id]['s'] = ss 
    return average_paramaters

def get_average_depth_map(plane_ids, plane_seg, average_plane_info):
    '''
    description: get the depth from the average parameters(p, q, r, s)
    parameter: the plane ids, plane_segs, the average plane infos
    return: evaluated depth maps of all planes
    '''
    depth_map = torch.zeros(plane_seg.size(), dtype = float)
    batch_size = len(depth_map)
    for i in range(batch_size):
        for the_id in plane_ids[i]: 
            the_id = int(the_id) 
            p = average_plane_info[i][the_id]['p']
            q = average_plane_info[i][the_id]['q']
            r = average_plane_info[i][the_id]['r']
            s = average_plane_info[i][the_id]['s']
            for v in range(len(depth_map[i][0])):
                for u in range(len(depth_map[i][0][v])): 
                    if int(plane_seg[i][0][v][u]) == the_id:
                        depth_map[i][0][v][u] = 1 / ((p * u + q * v + r) * s)
    return depth_map

def set_average_plane_info(plane_ids, plane_seg, average_plane_info):
    '''
    description: set the per pixel plane info to the average
    parameter: the plane ids, plane_segs, the average plane infos, the shape of the depth map
    return: average per pixel plane info
    '''
    p = torch.zeros(plane_seg.size(), dtype = float)
    q = torch.zeros(plane_seg.size(), dtype = float)
    r = torch.zeros(plane_seg.size(), dtype = float)
    s = torch.zeros(plane_seg.size(), dtype = float)
    batch_size = len(plane_seg)
    for i in range(batch_size):
        for the_id in plane_ids[i]:
            the_id = int(the_id) 
            pp = average_plane_info[i][the_id]['p']
            qq = average_plane_info[i][the_id]['q']
            rr = average_plane_info[i][the_id]['r']
            ss = average_plane_info[i][the_id]['s']
            for v in range(len(p[i][0])):
                for u in range(len(p[i][0][v])): 
                    if int(plane_seg[i][0][v][u]) == the_id:
                        p[i][0][v][u] = pp
                        q[i][0][v][u] = qq
                        r[i][0][v][u] = rr
                        s[i][0][v][u] = ss
    parameters = torch.cat((p, q, r, s), dim = 1)
    return parameters

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
    plane_ids = get_plane_ids(plane_seg) 
    print(depth_map_original.size(), plane_seg.size())

    parameters = get_parameter(depth_map_original, plane_seg)
    depth_map = get_depth_map(parameters)
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



    plane_info = get_average_plane_info(parameters, plane_seg, plane_ids)
    depth_average = get_average_depth_map(plane_ids, plane_seg, plane_info)

    min1 = 114514
    for i in range(len(plane_ids)):
        for the_id in plane_ids[i]:
            count = 0
            diff = 0
            for v in range(len(depth_map[i][0])):
                for u in range(len(depth_map[i][0][v])): 
                    if int(plane_seg[i][0][v][u]) == the_id:
                        count += 1
                        diff += abs(float(depth_average[i][0][v][u]) - float(depth_map_original[i][0][v][u]))
            diff /= count 
            print(i, the_id, count, diff)

#utils_test()