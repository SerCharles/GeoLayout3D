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
    for v in range(len(depth_map[0])):
        for u in range(len(depth_map[0][v])): 
            zz = float(depth_map[0][v][u]) 

            if u != len(depth_map[0][v]) - 1 and int(plane_seg[0][v][u + 1]) == int(plane_seg[0][v][u]):
                pp = 1.0 / float(depth_map[0][v][u + 1]) - 1.0 / float(depth_map[0][v][u])
            else: 
                pp = 1.0 / float(depth_map[0][v][u]) - 1.0 / float(depth_map[0][v][u - 1])
            if v != len(depth_map[0]) - 1 and int(plane_seg[0][v + 1][u]) == int(plane_seg[0][v][u]):
                qq = 1.0 / float(depth_map[0][v + 1][u]) - 1.0 / float(depth_map[0][v][u])
            else: 
                qq = 1.0 / float(depth_map[0][v][u]) - 1.0 / float(depth_map[0][v - 1][u])
            rr = 1 / zz - pp * u - qq * v
            ss = sqrt(pp ** 2 + qq ** 2 + rr ** 2) 
            pp /= ss 
            qq /= ss 
            rr /= ss 
            p[0][v][u] = pp 
            q[0][v][u] = qq 
            r[0][v][u] = rr 
            s[0][v][u] = ss 
    return p, q, r, s 

def get_depth_map(p, q, r, s):
    '''
    description: get the depth map from the parameters(p, q, r, s)
    parameter: the (p, q, r, s) value of all pixels
    return: evaluated depth map
    '''
    depth_map = torch.zeros(p.size(), dtype = float) 
    for v in range(len(depth_map[0])):
        for u in range(len(depth_map[0][v])): 
            depth_map[0][v][u] = 1 / ((float(p[0][v][u]) * u + float(q[0][v][u]) * v + float(r[0][v][u])) * float(s[0][v][u]))
    return depth_map

def get_average_plane_info(parameters, plane_seg, plane_ids):
    '''
    description: get the average plane info 
    parameter: parameters per pixel, plane segmentation per pixel, the list of plane ids 
    return: average plane info
    '''
    average_paramater = {}
    for t_id in plane_ids: 
        average_paramater[t_id] = {'count': 0, 'p': 0.0, 'q': 0.0, 'r': 0.0, 's': 0.0}
    p, q, r, s = parameters
    for v in range(len(plane_seg[0])) :
        for u in range(len(plane_seg[0][v])): 
            the_seg = int(plane_seg[0][v][u]) 
            the_p = float(p[0][v][u])
            the_q = float(q[0][v][u])
            the_r = float(r[0][v][u])
            the_s = float(s[0][v][u])
            average_paramater[the_seg]['count'] += 1
            average_paramater[the_seg]['p'] += the_p
            average_paramater[the_seg]['q'] += the_q
            average_paramater[the_seg]['r'] += the_r
            average_paramater[the_seg]['s'] += the_s
            
    for the_id in plane_ids: 
        pp = average_paramater[the_id]['p'] / average_paramater[the_id]['count']
        qq = average_paramater[the_id]['q'] / average_paramater[the_id]['count']
        rr = average_paramater[the_id]['r'] / average_paramater[the_id]['count']      
        ss = average_paramater[the_id]['s'] / average_paramater[the_id]['count']

        average_paramater[the_id]['p'] = pp 
        average_paramater[the_id]['q'] = qq 
        average_paramater[the_id]['r'] = rr
        average_paramater[the_id]['s'] = ss 
    return average_paramater

def get_average_depth_map(plane_ids, plane_seg, average_plane_info):
    '''
    description: get the depth from the average parameters(p, q, r, s)
    parameter: the plane ids, plane_segs, the average plane infos
    return: evaluated depth maps of all planes
    '''
    depth_map = torch.zeros(plane_seg.size(), dtype = float)
    for the_id in plane_ids: 
        p = average_plane_info[the_id]['p']
        q = average_plane_info[the_id]['q']
        r = average_plane_info[the_id]['r']
        s = average_plane_info[the_id]['s']
        for v in range(len(depth_map[0])):
            for u in range(len(depth_map[0][v])): 
                if int(plane_seg[0][v][u]) == the_id:
                    depth_map[0][v][u] = 1 / ((p * u + q * v + r) * s)
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

    for the_id in plane_ids: 
        pp = average_plane_info[the_id]['p']
        qq = average_plane_info[the_id]['q']
        rr = average_plane_info[the_id]['r']
        ss = average_plane_info[the_id]['s']
        for v in range(len(p[0])):
            for u in range(len(p[0][v])): 
                if int(plane_seg[0][v][u]) == the_id:
                    p[0][v][u] = pp
                    q[0][v][u] = qq
                    r[0][v][u] = rr
                    s[0][v][u] = ss
    return p, q, r, s

'''
name = 'E:\\dataset\\geolayout\\validation\\layout_depth\\04cdd02138664b138f281bb5ad8b957f_i1_3_layout.png'
depth_map_original = Image.open(name).convert('I')
transform_depth = transforms.Compose([transforms.Resize([152, 114]), transforms.ToTensor()])
depth_map_original = transform_depth(depth_map_original)
name = 'E:\\dataset\\geolayout\\validation\\layout_seg\\04cdd02138664b138f281bb5ad8b957f_i1_3_seg.png'
plane_seg = Image.open(name).convert('I')
transform_seg = transforms.Compose([transforms.Resize([152, 114], interpolation = PIL.Image.NEAREST), transforms.ToTensor()])
plane_seg = transform_seg(plane_seg)
plane_ids = [1, 3, 4, 5]


p, q, r, s = get_parameter(depth_map_original, plane_seg)
depth_map = get_depth_map(p, q, r, s)
print(p.shape, q.shape, r.shape, s.shape) 
print(depth_map_original.shape, depth_map.shape)
rate = 0
for v in range(len(depth_map[0])):
    for u in range(len(depth_map[0][v])): 
        diff = abs(float(depth_map[0][v][u]) - float(depth_map_original[0][v][u]))
        if diff < 0.5:  
            rate += 1 
        else: 
            kebab = 0
rate /= (len(depth_map[0]) * len(depth_map[0][0]))
print(rate)


plane_info = get_average_plane_info((p, q, r, s), plane_seg, plane_ids)
print(plane_info)
depth_average = get_average_depth_map(plane_ids, plane_seg, plane_info)
print(depth_average)

for the_id in plane_ids:
    count = 0
    diff = 0
    for v in range(len(depth_map[0])):
        for u in range(len(depth_map[0][v])): 
            if int(plane_seg[0][v][u]) == the_id:
                count += 1
                diff += abs(float(depth_average[0][v][u]) - float(depth_map_original[0][v][u]))
    diff /= count 
    print(the_id, count, diff)
'''