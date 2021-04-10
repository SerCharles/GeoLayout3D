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
from utils.post_process import * 
from train_utils import *
from utils.post_process import *
from data.load_matterport import *

def process():
    args = init_args()
    device, dataset_validation, dataloader_validation, model = init_valid_model(args)
    model.eval()
    all_base_names = dataset_validation.get_valid_filenames()
    current_flag = 0

    rms_total = 0.0
    rel_total = 0.0 
    rlog10_total = 0.0 
    rate_1_total = 0.0 
    rate_2_total = 0.0 
    rate_3_total = 0.0
    accuracy_total = 0.0
    for i, (image, layout_depth, layout_seg, intrinsic) in enumerate(dataloader_validation):
        batch_size = len(image)
        image = image.to(device)
        layout_depth = layout_depth.to(device)
        layout_seg = layout_seg.to(device)
        intrinsic = intrinsic.to(device)
        base_names = all_base_names[current_flag : current_flag + batch_size]
        current_flag += batch_size

        with torch.no_grad():
            parameter = model(image)

            depth_mine = get_depth_map(device, parameter, args.epsilon)
            layout_seg_gt = layout_seg.cpu().numpy()
            rms, rel, rlog10, rate_1, rate_2, rate_3 = depth_metrics(depth_mine, layout_depth)
            print('-' * 100)
            print('Batch [{} / {}]'.format(i + 1, len(dataloader_validation)))
            print('rms: {:.3f}, rel: {:.3f}, log10: {:.3f}, delta1: {:.3f}, delta2: {:.3f}, delta3: {:.3f}' \
            .format(rms, rel, rlog10, rate_1, rate_2, rate_3))

            final_labels, unique_label_list, parameter_list, plane_infos, final_depths = \
                post_process(parameter, intrinsic, args.cluster_threshold)
            save_plane_results(args, base_names, final_depths, final_labels, plane_infos)

            final_depths = torch.from_numpy(final_depths).to(device)
            rms, rel, rlog10, rate_1, rate_2, rate_3 = depth_metrics(final_depths, layout_depth)
            accuracy = seg_metrics(unique_label_list, final_labels, layout_seg_gt)

            rms_total += rms 
            rel_total += rel 
            rlog10_total += rlog10 
            rate_1_total += rate_1
            rate_2_total += rate_2
            rate_3_total += rate_3 
            accuracy_total += accuracy

            print('rms: {:.3f}, rel: {:.3f}, log10: {:.3f}, delta1: {:.3f}, delta2: {:.3f}, delta3: {:.3f}, accuracy: {:.4f}' \
            .format(rms, rel, rlog10, rate_1, rate_2, rate_3, accuracy))
    rms_avg = rms_total / len(dataloader_validation)
    rel_avg = rel_total / len(dataloader_validation)
    log10_avg = log10_total / len(dataloader_validation)
    rate_1_avg = rate_1_total / len(dataloader_validation)
    rate_2_avg = rate_2_total / len(dataloader_validation)
    rate_3_avg = rate_3_total / len(dataloader_validation)
    accuracy_avg = accuracy_total / len(dataloader_validation)
    print("*" * 100)
    print('rms: {:.3f}, rel: {:.3f}, log10: {:.3f}, delta1: {:.3f}, delta2: {:.3f}, delta3: {:.3f}, accuracy: {:.4f}' \
    .format(rms_avg, rel_avg, rlog10_avg, rate_1_avg, rate_2_avg, rate_3_avg, accuracy_avg))

process()