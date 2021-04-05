import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import torch
from torchvision import transforms

'''
def mean_shift_clustering(the_parameter_image):
    ''' 
    description: mean shift clustering algorithm
    parameter: the parameters of the image 
    return: the labels of all the images, the average parameter of all planes
    '''
    length_total = the_parameter_image
    threshold = length_total * 0.01
    bandwidth = estimate_bandwidth(data, quantile = 0.2, n_samples = 1000)
    print(bandwidth)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(data)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters = len(labels_unique)
    useful_cluster_list = []
    useful_average_parameters = []

    for i in range(n_clusters):
        equal_mask = np.equal(labels_unique[i])
        num_per_cluster[i] = np.sum(equal_mask)
        if num_per_cluster[i] > threshold:
            useful_cluster_list.append(labels_unique[i])
            sum_parameter = (the_parameter_image * equal_mask) / num_per_cluster[i]
            useful_average_parameters.append(sum_parameter)
    return labels, useful_cluster_list, useful_average_parameters
'''




