import numpy as np
import os
from math import *
from PIL import Image
import PIL
import torch
from torchvision import transforms
from get_parameter_geolayout import *

def log10(x):
    return torch.log(x) / log(10)

def depth_metrics(depth_map, depth_map_gt):
    '''
    description: get the depth metrics of the got depth map and the ground truth
    parameter: depth map of mine and the ground truth
    return: several metrics, rms, rel, log10, 1.25, 1.25^2, 1.25^3
    '''
    
    abs_diff = (depth_map - depth_map_gt).abs()

    mse = float((torch.pow(abs_diff, 2)).mean())
    rms = sqrt(mse)
    rlog10 = float((log10(depth_map) - log10(depth_map_gt)).abs().mean())
    rel = float((abs_diff / depth_map_gt).mean())

    max_ratio = torch.max(depth_map / depth_map_gt, depth_map_gt / depth_map)
    rate_1 = float((max_ratio < 1.25).float().mean())
    rate_2 = float((max_ratio < 1.25 ** 2).float().mean())
    rate_3 = float((max_ratio < 1.25 ** 3).float().mean())

    return rms, rel, rlog10, rate_1, rate_2, rate_3



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

rms, rel, rlog10, rate_1, rate_2, rate_3 = depth_metrics(depth_average, depth_map)
print(rms, rel, rlog10, rate_1, rate_2, rate_3)
'''