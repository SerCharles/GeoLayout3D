import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import torch
from math import *
from torchvision import transforms
import pandas as pd
import PIL
from PIL import Image
from utils.get_parameter_geolayout import *
from utils.evaluation_metrics import *


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
    #print(inconsistent_count)
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

def get_info(parameters, final_labels, picture_shape):
    ''' 
    description: get the final plane info and depth info of one picture, the shape of one picture
    parameter: the parameter of the picture, the final label of the picture pixels
    return: unique labels, plane infos, final depth info
    '''
    width = picture_shape[1]
    unique_labels = np.unique(final_labels)
    index = np.arange(len(final_labels), dtype = int)
    v = index // width 
    u = index % width 
    final_depth_list = []
    plane_info_list = []
    for label in unique_labels:
        mask_equal = np.equal(final_labels, label)
        parameter_current = parameters * mask_equal
        count_current = np.sum(mask_equal)
        sum_current = np.sum(parameter_current, axis = 1)
        avg_current = sum_current / count_current
        plane_info_list.append(avg_current)
        p = avg_current[0]
        q = avg_current[1]
        r = avg_current[2]
        s = avg_current[3]

        depth_frac = (p * u + q * v + r) * s
        depth_zero_mask = np.equal(depth_frac, 0)
        depth_frac = depth_frac + depth_zero_mask
        depth = 1 / depth_frac
        current_depth = (depth * mask_equal).reshape(1, -1)

        final_depth_list.append(current_depth)
    final_depth = np.concatenate(final_depth_list, axis = 0)
    final_depth = np.sum(final_depth, axis = 0)
    final_depth = final_depth.reshape(1, picture_shape[0], picture_shape[1])
    return unique_labels, np.array(plane_info_list), final_depth


def post_process(batch_result, threshold_ratio):
    '''
    description: post process one batch result
    parameter: the batch result, threshold ratio of useful cluster(min pixel count / total pixel count)
    return: the final labels, the unique labels, depth info of the picture, the planes
    '''
    batch_result = batch_result.detach()
    batch_size = len(batch_result)
    unique_label_list = []
    final_label_list = []
    plane_info_list = []
    final_depth_list = []
    for i in range(batch_size):
        parameter = batch_result[i].numpy()
        shape = parameter[0].shape 
        parameter = parameter.reshape(4, -1)
        #parameter = parameter.T
        final_labels = post_process_one(parameter, shape, threshold_ratio)
        unique_labels, plane_info, final_depth = get_info(parameter, final_labels, shape)


        final_depth = np.expand_dims(final_depth, axis = 0)
        final_labels = final_labels.reshape(1, 1, shape[0], shape[1])

        unique_label_list.append(unique_labels)
        plane_info_list.append(plane_info)
        final_depth_list.append(final_depth)
        final_label_list.append(final_labels)

    final_depth = np.concatenate(final_depth_list, axis = 0)
    final_labels = np.concatenate(final_label_list, axis = 0)
    return final_labels, unique_label_list, plane_info_list, final_depth




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
    final_labels, unique_labels, plane_info, final_depth = post_process(parameters, 0.01)
    final_depth = torch.from_numpy(final_depth)
    final_depth = final_depth.to(device) 
    rms, rel, rlog10, rate_1, rate_2, rate_3 = depth_metrics(final_depth, depth_map_original)
    print(rms, rel, rlog10, rate_1, rate_2, rate_3)

    plane_seg = plane_seg.detach().numpy()
    accuracy = seg_metrics(unique_labels, final_labels, plane_seg)
    print(accuracy)

#post_test()






