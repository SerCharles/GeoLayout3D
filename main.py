import numpy as np
import time
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from data.load_matterport import *
import models.senet as senet 
import models.modules as modules 
import models.net as net
from utils.loss_geolayout import * 
from train_utils import *
from utils.get_parameter_geolayout import *
from train import * 
from valid import * 

def main():
    args = init_args()
    device, dataset_training, dataset_validation, model, optimizer, start_epoch = init_model(args)
    #valid(args, device, dataset_validation, model, 0)
    for i in range(start_epoch, args.epochs):
        model = train(args, device, dataset_training, model, optimizer, i)
        valid(args, device, dataset_validation, model, i)

if __name__ == "__main__":
    main()