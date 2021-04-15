'''
Plot the train and test curve
'''

import matplotlib.pyplot as plt
import os
import numpy as np
from tensorboardX import SummaryWriter

from global_utils import *

class PlotCurves:
    def get_info(self):
        '''
        description: get the train and test loss info from the log
        parameter: empty
        return: the average train and test loss info
        '''
        file_name = os.path.join(args.save_dir, args.cur_name, 'log.txt')

        train_loss_param = []
        train_loss_dis = []
        train_loss_depth = []
        valid_loss_param = []
        valid_loss_dis = []
        valid_loss_depth = []
        for i in range(args.epochs):
            train_loss_param.append([])
            train_loss_dis.append([])
            train_loss_depth.append([])
            valid_loss_param.append([])
            valid_loss_dis.append([])
            valid_loss_depth.append([])

        with open(file_name) as f:
            lines = f.readlines()
            for i in range(len(lines)):
                words = lines[i].split()
                if(len(words) <= 0):
                    continue
                if words[0] == 'Train:':
                    epoch = int(words[2][1:]) - 1
                    loss_line = lines[i + 1].split()
                    loss_param = float(loss_line[4][ : -1])
                    loss_dis = float(loss_line[7][ : -1])
                    loss_depth = float(loss_line[10][ : ])
                    train_loss_param[epoch].append(loss_param)
                    train_loss_dis[epoch].append(loss_dis)
                    train_loss_depth[epoch].append(loss_depth)
                elif words[0] == 'Valid:':
                    epoch = int(words[2][1:]) - 1
                    loss_line = lines[i + 1].split()
                    loss_param = float(loss_line[4][ : -1])
                    loss_dis = float(loss_line[7][ : -1])
                    loss_depth = float(loss_line[10][ : -1])
                    valid_loss_param[epoch].append(loss_param)
                    valid_loss_dis[epoch].append(loss_dis)
                    valid_loss_depth[epoch].append(loss_depth)

        train_loss_param = np.mean(np.array(train_loss_param), axis = 1)
        train_loss_dis = np.mean(np.array(train_loss_dis), axis = 1)
        train_loss_depth = np.mean(np.array(train_loss_depth), axis = 1)
        valid_loss_param = np.mean(np.array(valid_loss_param), axis = 1)
        valid_loss_dis = np.mean(np.array(valid_loss_dis), axis = 1)
        valid_loss_depth = np.mean(np.array(valid_loss_depth), axis = 1)
        train_loss = train_loss_param + train_loss_dis + train_loss_depth
        valid_loss = valid_loss_param + valid_loss_dis + valid_loss_depth 

        return train_loss, train_loss_param, train_loss_dis, train_loss_depth, valid_loss, valid_loss_param, valid_loss_dis, valid_loss_depth         


    def plot_all(self, train_loss, train_loss_param, train_loss_dis, train_loss_depth, valid_loss, valid_loss_param, valid_loss_dis, valid_loss_depth):
        '''
        description: plot all the loss info on one chart
        parameter: all the losses 
        return: empty
        '''
        writer = SummaryWriter()
        for epoch in range(len(train_loss)):
            writer.add_scalars('scalar/test', {"train_loss" : train_loss[epoch], "valid_loss" : valid_loss[epoch], \
                'train_loss_param' : train_loss_param[epoch], 'train_loss_dis' : train_loss_dis[epoch], 'train_loss_depth' : train_loss_depth[epoch], \
                'valid_loss_param' : valid_loss_param[epoch], 'valid_loss_dis' : valid_loss_dis[epoch], 'valid_loss_depth' : valid_loss_depth[epoch]}, epoch)
        writer.close()
        
    def plot_one(self, train_loss, valid_loss):
        '''
        description: plot one kind of train/valid loss on the chart
        parameter: train and valid loss
        return: empty
        '''
        writer = SummaryWriter()
        for epoch in range(len(train_loss)):
            writer.add_scalars('scalar/test', {"train" : train_loss[epoch], "valid" : valid_loss[epoch]}, epoch)
        writer.close()



    def __init__(self, args):
        ''' 
        description: the main function of plotting
        parameter: args
        return: empty
        '''
        super(PlotCurves, self).__init__()
        self.args = args
        train_loss, train_loss_param, train_loss_dis, train_loss_depth, valid_loss, valid_loss_param, valid_loss_dis, valid_loss_depth = self.get_info()
        while(True):
            print('Please input the type you want to plot')
            print('0: Plot all 4 types of loss of training and testing')
            print('1: Plot only the total loss')
            print('2: Plot only the parameter loss')
            print('3: Plot only the discrimitive loss')
            print('4: Plot only the depth loss')
            try:
                the_type = int(input())
            except: 
                print('Invalid input, please try again!')
                continue 
            if the_type == 0: 
                self.plot_all(train_loss, train_loss_param, train_loss_dis, train_loss_depth, valid_loss, valid_loss_param, valid_loss_dis, valid_loss_depth)
                break 
            elif the_type == 1: 
                self.plot_one(train_loss, valid_loss)
                break
            elif the_type == 2: 
                self.plot_one(train_loss_param, valid_loss_param)
                break
            elif the_type == 3: 
                self.plot_one(train_loss_dis, valid_loss_dis)
                break
            elif the_type == 4: 
                self.plot_one(train_loss_depth, valid_loss_depth)
                break
            else: 
                print('Invalid input, please try again!')

if __name__ == "__main__":
    args = init_args()
    a = PlotCurves(args)