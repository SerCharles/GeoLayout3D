import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import torch
from math import *
from torchvision import transforms


def mean_shift_clustering(the_parameter_image):
    ''' 
    description: mean shift clustering algorithm
    parameter: the parameters of the image 
    return: the labels of all the pixels
    '''
    length_total = the_parameter_image
    threshold = length_total * 0.01
    bandwidth = estimate_bandwidth(data, quantile = 0.2, n_samples = 1000)
    print(bandwidth)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(data)
    labels = ms.labels_
    return labels

def get_average_planes(the_parameter, the_labels):
    ''' 
    description: get the average parameter of all planes based on the segmentation labels
    parameter: the parameters of the image, the segmentation labels
    return: the useful plane ids and their parameters
    '''
    labels_unique = np.unique(the_labels)
    n_clusters = len(labels_unique)
    useful_cluster_list = []
    useful_average_parameters = []

    for i in range(n_clusters):
        equal_mask = np.equal(labels_unique[i])
        the_num = np.sum(equal_mask)
        if the_num > threshold:
            useful_cluster_list.append(labels_unique[i])
            sum_parameter = (the_parameter * equal_mask) / the_num
            useful_average_parameters.append(sum_parameter)
    return useful_cluster_list, useful_average_parameters


def get_labels_per_pixel(width, the_labels, useful_cluster_list, useful_average_parameters):
    '''
    description: get the pixel labels based on the depth we got
    parameter: the width of the image, 
    return: the renewed label, the inconsistent count
    '''
    inconsistent_count = 0
    for i in range(len(the_labels)):
        v = i / width 
        u = i % width 
        min_positive_depth = inf 
        the_label = -1
        for j in range(len(useful_cluster_list)):
            p = useful_average_parameters[j][0]
            q = useful_average_parameters[j][1]
            r = useful_average_parameters[j][2]
            s = useful_average_parameters[j][3]
            depth = 1 / ((p * u + q * v + r) * s)
            if depth < min_positive_depth and depth >= 0: 
                min_positive_depth = depth 
                the_label = useful_cluster_list[j]
        if the_label != the_labels[i]:
            the_labels[i] = the_label
            inconsistent_count += 1 
    return the_labels, inconsistent_count

'''
def post_process_one(parameter_info):
    '''
    description: post process one picture, renewing its label and depth info 
    parameter: the parameter of the picture
    return: the label, depth info of the picture, the planes
    '''
    raw_labels = mean_shift_clustering(parameter_info)
    while(True):
'''



