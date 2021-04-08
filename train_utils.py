import argparse
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
import os
from data.load_matterport import *
import models.senet as senet 
import models.modules as modules 
import models.net as net
import PIL 
from PIL import Image


def init_args():
    '''
    description: load train args
    parameter: empty
    return: args
    '''
    parser = argparse.ArgumentParser(description = 'PyTorch GeoLayout3D Training')
    parser.add_argument('--seed', default = 1453)
    parser.add_argument('--cuda',  type = int, default = 1, help = 'use GPU or not')
    parser.add_argument('--gpu_id', type = int, default = 1, help = 'GPU device id used')
    parser.add_argument('--epochs', default = 200, type = int)
    parser.add_argument('--start_epoch', default = 0, type = int,
                    help = 'manual epoch number (useful on restarts)')
    parser.add_argument('--learning_rate', '--lr', default = 1e-4, type = float)
    parser.add_argument('--weight_decay', '--wd',  default = 1e-4, type = float)
    parser.add_argument('--epsilon', default = 1e-8, type = float)
    parser.add_argument('--batch_size', '--bs', default = 4, type = int)
    parser.add_argument('--delta_v', default = 0.1, type = float)
    parser.add_argument('--delta_d', default = 1.0, type = float)
    parser.add_argument('--alpha', default = 0.5, type = float)
    parser.add_argument('--beta', default = 1.0, type = float)
    parser.add_argument('--data_dir', default = '/home/shenguanlin/geolayout', type = str)
    parser.add_argument('--save_dir', default = '/home/shenguanlin/geolayout_result', type = str)
    parser.add_argument('--cur_name', default = 'test', type = str)

    args = parser.parse_args()
    return args

def adjust_learning_rate(args, optimizer, epoch):
    '''
    description: reduce the learning rate to 10% every 50 epoch
    parameter: args, optimizer, epoch_num
    return: empty
    '''
    lr = args.learning_rate * (0.1 ** (epoch // 50))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(args, state, epoch):
    '''
    description: save the checkpoint
    parameter: args, optimizer, epoch_num
    return: empty
    '''
    file_dir = os.path.join(args.save_dir, args.cur_name)
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    if epoch % 5 == 0:
        filename = os.path.join(file_dir, 'checkpoint_' + str(epoch) + '.pth')
        torch.save(state, filename)

def write_log(args, epoch, batch, the_type, info):
    '''
    description: write the log file
    parameter: args, epoch, type(training/validation/testing), the info you want to write
    return: empty
    '''
    file_dir = os.path.join(args.save_dir, args.cur_name)
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    file_name = os.path.join(file_dir, 'log.txt')
    if epoch == 0 and batch == 0 and the_type == 'training':
        file = open(file_name, 'w')
        file.close()
    file = open(file_name, 'a')
    file.write(info + '\n')
    file.close()

def init_model(args):
    '''
    description: init the device, dataloader, model, optimizer of the model
    parameter: args
    return: device, dataloader_train, dataloader_valid, model, optimizer
    '''
    print(args)
    print('getting device...', end='')
    torch.manual_seed(args.seed)
    if args.cuda == 1:
        torch.cuda.set_device(args.gpu_id)
        device = torch.device(args.gpu_id)
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    print(device)

    print('Initialize model')
    
    original_model = senet.senet154(pretrained = 'imagenet')
    Encoder = modules.E_senet(original_model)
    model = net.model(Encoder, num_features = 2048, block_channel = [256, 512, 1024, 2048])

    if(args.start_epoch != 0):
        file_dir = os.path.join(args.save_dir, args.cur_name)
        filename = os.path.join(file_dir, 'checkpoint_' + str(args.start_epoch) + '.pth')
        model.load_state_dict(torch.load(filename))

    if device:
        model.to(device)

    print('Getting optimizer')
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay = args.weight_decay)


    print('Getting dataset')
    dataset_training = MatterPortDataSet(args.data_dir, 'training')
    dataset_validation = MatterPortDataSet(args.data_dir, 'validation')
    dataloader_training =  DataLoader(dataset_training, batch_size = args.batch_size, shuffle = True, num_workers = 5)
    dataloader_validation = DataLoader(dataset_validation, batch_size = args.batch_size, shuffle = False, num_workers = 5)
    print('Data got!')

    return device, dataloader_training, dataloader_validation, model, optimizer, args.start_epoch

def save_plane_results(args, base_names, final_depths, final_labels, plane_infos):
    ''' 
    description: save the plane results
    parameter: args, the file base names of the batch, the final depth, final segmentation and final plane infos
    return: empty
    '''
    depth_dir = os.path.join(args.save_dir, args.cur_name, 'depth')
    seg_dir = os.path.join(args.save_dir, args.cur_name, 'seg')
    plane_dir = os.path.join(args.save_dir, args.cur_name, 'plane')
    if not os.path.exists(depth_dir): 
        os.mkdir(depth_dir)
    if not os.path.exists(seg_dir): 
        os.mkdir(seg_dir)
    if not os.path.exists(plane_dir): 
        os.mkdir(plane_dir)

    batch_size = len(base_names)
    for i in range(batch_size):
        base_name = base_names[i]
        depth = final_depths[i]
        seg = final_labels[i]
        plane_info = np.array(plane_infos[i])
        depth = depth * 4000
        depth = depth.astype(int)

        depth_image = Image.fromarray(depth).convert('I')
        seg_image = Image.fromarray(seg).convert('I')

        depth_name = os.path.join(depth_dir, base_name, '_depth.png')
        seg_name = os.path.join(depth_dir, base_name, '_plane.png')
        plane_name = os.path.join(depth_dir, base_name, '_plane.npy')

        depth_image.save(depth_name)
        seg_image.save(seg_name)
        np.save(plane_name, plane_info)
    

