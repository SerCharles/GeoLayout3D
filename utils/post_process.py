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

def get_plane_info(parameter_list, intrinsics):
    ''' 
    description: calculate the plane infos based on the parameters and intrinsics
    parameter: the list of plane parameters, the intrinsics
    return: plane infos
    '''
    batch_size = len(intrinsics)
    plane_info = []
    for i in range(batch_size):
        intrinsic = intrinsics[i]
        parameter = parameter_list[i]
        fx = intrinsic[0][0]
        fy = intrinsic[1][1]
        u0 = intrinsic[2][0]
        v0 = intrinsic[2][1]
        planes = []
        for j in range(len(parameter)):
            p = parameter[j][0]
            q = parameter[j][1]
            r = parameter[j][2]
            s = parameter[j][3]
            a = fx * p * s 
            b = fy * q * s 
            c = p * s * u0 + q * s * v0 + r * s 
            d = -1
            plane = np.array([a, b, c, d])
            norm = np.linalg.norm(plane)
            plane = plane / norm 
            planes.append(plane)
        plane_info.append(planes)
    return plane_info







def post_process(batch_result, intrinsics, threshold_ratio):
    '''
    description: post process one batch result
    parameter: the batch result, intrinsics, threshold ratio of useful cluster(min pixel count / total pixel count)
    return: the final labels, the unique labels, the parameters of the planes, depth info of the picture
    '''
    batch_result = batch_result.cpu()
    intrinsics = intrinsics.cpu().numpy()
    batch_size = len(batch_result)
    unique_label_list = []
    final_label_list = []
    parameter_list = []
    final_depth_list = []
    for i in range(batch_size):
        parameter = batch_result[i].numpy()
        shape = parameter[0].shape 
        parameter = parameter.reshape(4, -1)
        #parameter = parameter.T
        final_labels = post_process_one(parameter, shape, threshold_ratio)
        unique_labels, avg_parameter, final_depth = get_info(parameter, final_labels, shape)


        final_depth = np.expand_dims(final_depth, axis = 0)
        final_labels = final_labels.reshape(1, 1, shape[0], shape[1])

        unique_label_list.append(unique_labels)
        parameter_list.append(avg_parameter)
        final_depth_list.append(final_depth)
        final_label_list.append(final_labels)

    final_depth = np.concatenate(final_depth_list, axis = 0)
    final_labels = np.concatenate(final_label_list, axis = 0)
    plane_info = get_plane_info(parameter_list, intrinsics)
    return final_labels, unique_label_list, parameter_list, plane_info, final_depth




def post_test():
    transform_depth = transforms.Compose([transforms.Resize([152, 114]), transforms.ToTensor()])
    transform_seg = transforms.Compose([transforms.Resize([152, 114], interpolation = PIL.Image.NEAREST), transforms.ToTensor()])
    name = 'E:\\dataset\\geolayout\\training\\layout_depth\\00ebbf3782c64d74aaf7dd39cd561175_i1_0_layout.png'
    depth_map_original_0 = Image.open(name).convert('I')
    depth_map_original_0 = transform_depth(depth_map_original_0) / 4000.0
    name = 'E:\\dataset\\geolayout\\training\\layout_depth\\00ebbf3782c64d74aaf7dd39cd561175_i2_0_layout.png' 
    depth_map_original_1 = Image.open(name).convert('I')
    depth_map_original_1 = transform_depth(depth_map_original_1) / 4000.0
    depth_map_original = torch.stack((depth_map_original_0, depth_map_original_1))
    name = 'E:\\dataset\\geolayout\\training\\layout_seg\\00ebbf3782c64d74aaf7dd39cd561175_i1_0_seg.png'
    plane_seg_0 = Image.open(name).convert('I')
    plane_seg_0 = transform_seg(plane_seg_0)
    name = 'E:\\dataset\\geolayout\\training\\layout_seg\\00ebbf3782c64d74aaf7dd39cd561175_i2_0_seg.png'
    plane_seg_1 = Image.open(name).convert('I')
    plane_seg_1 = transform_seg(plane_seg_1)
    plane_seg = torch.stack((plane_seg_0, plane_seg_1)) 
    plane_ids = get_plane_ids(plane_seg) 

    intrinsic_0 = torch.tensor([[1.0755e+03, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 1.0757e+03, 0.0000e+00],
        [6.2730e+02, 5.0792e+02, 1.0000e+00]])
    intrinsic_1 = torch.tensor([[1.07461e+03, 0.00000e+00, 0.00000e+00],
        [0.00000e+00, 1.07493e+03, 0.00000e+00],
        [6.30692e+02, 5.23124e+02, 1.00000e+00]])
    intrinsics= torch.stack((intrinsic_0, intrinsic_1)) 


    device = torch.device("cpu")
    parameters = get_parameter(device, depth_map_original, plane_seg, 1e-8)
    final_labels, unique_labels, parameters, plane_info, final_depth = post_process(parameters, intrinsics, 0.01)
    final_depth = torch.from_numpy(final_depth)
    final_depth = final_depth.to(device) 
    rms, rel, rlog10, rate_1, rate_2, rate_3 = depth_metrics(final_depth, depth_map_original)
    print(rms, rel, rlog10, rate_1, rate_2, rate_3)

    plane_seg = plane_seg.detach().numpy()
    accuracy = seg_metrics(unique_labels, final_labels, plane_seg)
    print(accuracy)
    print(plane_info)

#post_test()






