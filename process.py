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
from post_process import *
from data.load_matterport import *

def process():
    args = init_args()
    device, dataset_validation, dataloader_validation, model = init_valid_model(args)
    model.eval()
    all_base_names = dataset_validation.get_valid_filenames()
    current_flag = 0

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
            final_labels, unique_label_list, parameter_list, plane_infos, final_depths = \
                post_process(parameter, intrinsic, args.cluster_threshold)
            save_plane_results(args, base_names, final_depths, final_labels, plane_infos)