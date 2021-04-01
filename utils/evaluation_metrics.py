import numpy as np
import os
from math import *
from PIL import Image
import PIL
import torch
from torchvision import transforms
from utils.get_parameter_geolayout import *

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
    aa = log10(depth_map)
    bb = log10(depth_map_gt)
    cc = (aa - bb).abs()
    rlog10 = float(cc.mean())
    rel = float((abs_diff / depth_map_gt).mean())

    max_ratio = torch.max(depth_map / depth_map_gt, depth_map_gt / depth_map)
    rate_1 = float((max_ratio < 1.25).float().mean())
    rate_2 = float((max_ratio < 1.25 ** 2).float().mean())
    rate_3 = float((max_ratio < 1.25 ** 3).float().mean())

    return rms, rel, rlog10, rate_1, rate_2, rate_3






#unit test code
def metrics_test():
    transform_depth = transforms.Compose([transforms.Resize([152, 114]), transforms.ToTensor()])
    transform_seg = transforms.Compose([transforms.Resize([152, 114], interpolation = PIL.Image.NEAREST), transforms.ToTensor()])
    name = 'E:\\dataset\\geolayout\\validation\\layout_depth\\04cdd02138664b138f281bb5ad8b957f_i1_3_layout.png'
    depth_map_original_0 = Image.open(name).convert('I')
    depth_map_original_0 = transform_depth(depth_map_original_0) / 4000.0
    name = 'E:\\dataset\\geolayout\\validation\\layout_depth\\075307518bc2495498609ee2ff6dd003_i1_2_layout.png'
    depth_map_original_1 = Image.open(name).convert('I')
    depth_map_original_1 = transform_depth(depth_map_original_1) / 4000.0
    depth_map_original = torch.stack((depth_map_original_0, depth_map_original_0))
    name = 'E:\\dataset\\geolayout\\validation\\layout_seg\\04cdd02138664b138f281bb5ad8b957f_i1_3_seg.png'
    plane_seg_0 = Image.open(name).convert('I')
    plane_seg_0 = transform_seg(plane_seg_0)
    name = 'E:\\dataset\\geolayout\\validation\\layout_seg\\075307518bc2495498609ee2ff6dd003_i1_2_seg.png'
    plane_seg_1 = Image.open(name).convert('I')
    plane_seg_1 = transform_depth(plane_seg_1)
    plane_seg = torch.stack((plane_seg_0, plane_seg_0))  
    print(depth_map_original.size(), plane_seg.size())


    plane_ids = get_plane_ids(plane_seg)

    parameters = get_parameter(depth_map_original, plane_seg)
    depth_map = get_depth_map(parameters)
    plane_info = get_average_plane_info(parameters, plane_seg, plane_ids)
    depth_average = get_average_depth_map(plane_ids, plane_seg, plane_info)

    rms, rel, rlog10, rate_1, rate_2, rate_3 = depth_metrics(depth_average, depth_map)
    print(rms, rel, rlog10, rate_1, rate_2, rate_3)
#metrics_test()