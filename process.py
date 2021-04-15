''' 
After training the network, use this to conduct the clustering, iterative improvement and visualization
'''


import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import torch
from math import *
from torchvision import transforms
import pandas as pd
import PIL
from PIL import Image
from utils.utils import *
from utils.metrics import *
from utils.post_process import * 
from global_utils import *
from utils.post_process import *
from data.dataset import *


class AverageMeter(object):
    ''' 
    used in calculating the average depth metrics
    '''

    def __init__(self, args):
        self.num = 0
        self.rms = 0.0
        self.rel = 0.0 
        self.rlog10 = 0.0 
        self.rate_1 = 0.0 
        self.rate_2 = 0.0 
        self.rate_3 = 0.0
        file_dir = os.path.join(args.save_dir, args.cur_name)
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)
        file_name = os.path.join(file_dir, 'valid_log.txt')
        self.log_name = file_name
        file = open(self.log_name, 'w')
        file.close()
    
    def add_one(self, metrics):
        rms, rel, rlog10, rate_1, rate_2, rate_3 = metrics 
        self.num += 1
        self.rms += rms
        self.rel += rel
        self.rlog10 += rlog10
        self.rate_1 += rate_1
        self.rate_2 += rate_2 
        self.rate_3 += rate_3 
        result_string = 'rms: {:.3f}, rel: {:.3f}, log10: {:.3f}, delta1: {:.3f}, delta2: {:.3f}, delta3: {:.3f}' \
            .format(rms, rel, rlog10, rate_1, rate_2, rate_3)
        print(result_string)
        file = open(self.log_name, 'a')
        file.write(result_string + '\n')
        file.close()
    
    def show_average(self):
        rms = self.rms / self.num
        rel = self.rel / self.num
        rlog10 = self.rlog10 / self.num
        rate_1 = self.rate_1 / self.num
        rate_2 = self.rate_2 / self.num
        rate_3 = self.rate_3 / self.num
        result_string = 'rms: {:.3f}, rel: {:.3f}, log10: {:.3f}, delta1: {:.3f}, delta2: {:.3f}, delta3: {:.3f}' \
            .format(rms, rel, rlog10, rate_1, rate_2, rate_3)
        print(result_string)
        file = open(self.log_name, 'a')
        file.write(result_string + '\n')
        file.close()

    def print_info(self, print_string):
        print(print_string)
        file = open(self.log_name, 'a')
        file.write(print_string + '\n')
        file.close()

def process():
    args = init_args()
    device, dataset_validation, dataloader_validation, model = init_valid_model(args)
    model.eval()
    all_base_names = dataset_validation.get_valid_filenames()
    current_flag = 0

    metrics_pixel = AverageMeter(args)
    metrics_avg = AverageMeter(args)
    metrics_final = AverageMeter(args)
    accuracy_total = 0.0
    for i, (image, layout_depth, layout_seg, intrinsic) in enumerate(dataloader_validation):
        batch_size = len(image)
        if device:
            image = image.cuda()
            layout_depth = layout_depth.cuda()
            layout_seg = layout_seg.cuda()
            intrinsic = intrinsic.cuda()
        base_names = all_base_names[current_flag : current_flag + batch_size]
        current_flag += batch_size

        with torch.no_grad():
            parameter = model(image)

            max_num = get_plane_max_num(layout_seg)
            average_plane_info = get_average_plane_info(device, parameter, layout_seg, max_num)
            parameter_gt = get_parameter(device, layout_depth, layout_seg, args.epsilon)
            average_depth = get_average_depth_map(device, layout_seg, average_plane_info, args.epsilon)
            depth_mine = get_depth_map(device, parameter, args.epsilon)


            metrics_pixel.print_info('-' * 100 + '\nBatch [{} / {}]'.format(i + 1, len(dataloader_validation)))

            metrics_pixel.add_one(depth_metrics(depth_mine, layout_depth))
            metrics_avg.add_one(depth_metrics(average_depth, layout_depth))

            layout_seg_gt = layout_seg.cpu().numpy()
            final_labels, unique_label_list, parameter_list, plane_infos, final_depths = \
                post_process(parameter, intrinsic, args.cluster_threshold)
            save_plane_results(args, base_names, final_depths, final_labels, plane_infos)

            final_depths = torch.from_numpy(final_depths).cuda()
            metrics_final.add_one(depth_metrics(final_depths, layout_depth))
            accuracy = seg_metrics(unique_label_list, final_labels, layout_seg_gt)

            accuracy_total += accuracy

    accuracy_avg = accuracy_total / len(dataloader_validation)
    metrics_pixel.print_info('-' * 100 + '\nAverage Info:')
    metrics_pixel.show_average()
    metrics_avg.show_average()
    metrics_final.show_average()
    print('accuracy: {:.4f}'.format(accuracy_avg))

if __name__ == '__main__':
    process()