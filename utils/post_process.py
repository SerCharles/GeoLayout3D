import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import torch
from math import *
from torchvision import transforms
import pandas as pd
import PIL
from PIL import Image
from get_parameter_geolayout import *


def mean_shift_clustering(the_parameter_image):
    ''' 
    description: mean shift clustering algorithm
    parameter: the parameters of the image 
    return: the labels of all the pixels
    '''
    bandwidth = estimate_bandwidth(the_parameter_image, quantile = 0.2, n_samples = 1000)
    #print(bandwidth)
    ms = MeanShift(bandwidth = bandwidth, bin_seeding = True)
    ms.fit(the_parameter_image)
    labels = ms.labels_
    return labels

def get_average_planes(the_parameter, the_labels, threshold_ratio):
    ''' 
    description: get the average parameter of all planes based on the segmentation labels
    parameter: the parameters of the image, the segmentation labels, threshold ratio of useful cluster(min pixel count / total pixel count)
    return: the useful plane ids and their parameters
    '''
    total_num = len(the_labels)
    threshold = total_num * threshold_ratio
    labels_unique = np.unique(the_labels)
    n_clusters = len(labels_unique)
    useful_cluster_list = []
    useful_average_parameters = []

    for i in range(n_clusters):
        equal_mask = np.equal(the_labels, labels_unique[i])
        the_num = np.sum(equal_mask)
        if the_num > threshold:
            useful_cluster_list.append(labels_unique[i])
            sum_parameter = np.sum(the_parameter * equal_mask, axis = 1)
            avg_parameter = sum_parameter / the_num
            useful_average_parameters.append(avg_parameter)
    return useful_cluster_list, useful_average_parameters


def get_labels_per_pixel(width, the_labels, useful_cluster_list, useful_average_parameters):
    '''
    description: get the pixel labels based on the depth we got
    parameter: the width of the image, the current labels, useful_cluster_list, useful_average_parameters 
    return: the renewed label, the inconsistent count
    '''
    current_max = 14530529
    index = np.arange(len(the_labels), dtype = int)
    v = index // width 
    u = index % width 
    depth_list = []
    for j in range(len(useful_cluster_list)):
        p = useful_average_parameters[j][0]
        q = useful_average_parameters[j][1]
        r = useful_average_parameters[j][2]
        s = useful_average_parameters[j][3]
        depth = 1. / ((p * u + q * v + r) * s)

        positive_mask = (depth >= 0)
        negative_mask = ~ positive_mask
        depth_positive = depth * positive_mask
        depth_negative = negative_mask * current_max
        depth = depth_positive + depth_negative

        depth = depth.reshape(1, depth.shape[0])
        depth_list.append(depth)
    depths = np.concatenate(depth_list, axis = 0)
    min_index = np.argmin(depths, axis = 0)
    np_cluster = np.array(useful_cluster_list, dtype = int)
    new_labels = np_cluster[min_index]
    inconsistent_count = np.sum(new_labels != the_labels)
    print(inconsistent_count)
    return new_labels, inconsistent_count


def post_process_one(parameter_info, picture_shape, threshold_ratio):
    '''
    description: post process one picture of result, renewing its label and depth info 
    parameter: the parameter of the picture, the shape of one picture, threshold ratio of useful cluster(min pixel count / total pixel count)
    return: the labels of the picture pixels
    '''
    parameter_info_pandas = pd.DataFrame(parameter_info.T, columns = list('pqrs'))
    width = picture_shape[1]
    labels = mean_shift_clustering(parameter_info_pandas) 

    max_iter = 1000
    for i in range(max_iter):
        useful_cluster_list, useful_average_parameters = get_average_planes(parameter_info, labels, threshold_ratio)
        labels, inconsistent_count = get_labels_per_pixel(width, labels, useful_cluster_list, useful_average_parameters)
        if inconsistent_count == 0:
            break 
    return labels
    
def post_process(batch_result, threshold_ratio):
    '''
    description: post process one batch result
    parameter: the batch result, threshold ratio of useful cluster(min pixel count / total pixel count)
    return: the label, depth info of the picture, the planes
    '''
    batch_result = batch_result.detach()
    batch_size = len(batch_result)
    for i in range(batch_size):
        parameter = batch_result[i].numpy()
        shape = parameter[0].shape 
        parameter = parameter.reshape(4, -1)
        #parameter = parameter.T
        final_labels = post_process_one(parameter, shape, threshold_ratio)
        


def post_test():
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
    parameters = get_parameter(device, depth_map_original, plane_seg, 1e-8)
    post_process(parameters, 0.01)
    '''
    depth_map = get_depth_map(device, parameters, 1e-8)
    max_num = get_plane_max_num(plane_seg)
    plane_info = get_average_plane_info(device, parameters, plane_seg, max_num)
    depth_average = get_average_depth_map(device, plane_seg, plane_info, 1e-8)
    parameters_avg = set_average_plane_info(device, plane_seg, plane_info)
    '''
post_test()






