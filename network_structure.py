from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import os
import argparse
from utils import progress_bar
import numpy as np
import h5py
from utils import *
from models.utils import get_model, get_criterion
from passers import Passer
from savers import save_weights
from loaders import *
from graph import *
import pickle as pkl 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# n = 'resnet'
n = 'lenet'
dataset = 'mnist'
# d = 'cifar10'
net = get_model(n,dataset)
net = net.to(device)
criterion = F.nll_loss

def get_network_structure():
    layers = net.named_modules()
    layer_info_dict = {}
    for name, layer in layers:
        tl = type(layer)
        if tl == type(nn.Conv2d(1,1,3)):
            param = [p for p in layer.parameters()][0].data.cpu().numpy()
            layer_info_dict[name] = {'name':'conv2d', 'in':layer.in_channels,'out':layer.out_channels, 'stride':layer.stride, 'padding':layer.padding, 'ks':layer.kernel_size,'weights':param}
        elif tl == type(nn.Linear(2,1)):
            param = [p for p in layer.parameters()][0].data.cpu().numpy()
            layer_info_dict[name] = {'name':'fc', 'in':layer.in_features,'out':layer.out_features, 'weights':param}
        else:
            pass
            print("Undefined",type(layer))
    return layer_info_dict

def calculate_sum(weights, level, total_level):
    weight_sum = 0
    weight_count = 0
    for w in weights:
        if level<total_level:
            s,c = calculate_sum(w, level+1, total_level)
            weight_sum += s
            weight_count += c
        else:
            weight_sum += w
            weight_count += 1
    return weight_sum, weight_count

def calculate_std(weights, mean, level, total_level):
    weight_std = 0
    for w in weights:
        if level<total_level:
            weight_std += calculate_std(w, mean, level+1, total_level)
        else:
            weight_std += (w-mean)*(w-mean)
    return weight_std

def transform_distribution(weights, mean, std, level, total_level):
    for i, w in enumerate(weights):
        if level<total_level:
            transform_distribution(w, mean, std, level+1, total_level)
        else:
            weights[i] = (w - mean)/std

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

for epoch in [1]:
    print('==> Loading checkpoint for epoch {}...'.format(epoch))
    checkpoint = torch.load('./scripts/checkpoint/lenet_mnist/ckpt_trial_0_epoch_' + str(epoch)+'.t7')
    net.load_state_dict(checkpoint['net'])
  
    functloader = loader('mnist_test', batch_size=100, subset=list(range(0, 1000)))
    passer = Passer(net, functloader, criterion, device)
    #print([f for f in functloader][0][0].shape)
    #layer_info_dict = get_network_structure()

    node_number = [0,0,0]   # the node number of the last layer (in three dimension)
    if 1==1:
        node_layer = []         # node number for each layer
        total_node_number = 0   # total node number 
        if dataset == 'mnist' or dataset == 'cifar10':
            node_number = [32, 32, 1]  # weight, height, channel of the input image
        else:
            node_number = [64, 64, 1]
        node_layer.append(node_number)
        total_node_number += node_number[0]*node_number[1]*node_number[2]        
        
        layer_info_dict = passer.get_network_structure()
        weights = []
        #### 1. initialize the adjacency matrix ( n by n, where n is the total neural numbers) ####
        # The aim is to get the total neural numbers and adjust the weight to obey the normal distribution
        # Note: Input image should also be included in order to calculate the weights
        # The matrix should look like as follows:
        #           | layer 1       |layer 2| layer 3       |layer 4|   
        # layer1    0       0       0.2     0       0       0
        #   -       0       0       0.6     0       0       0 
        # layer2    0.2     0.6     0       -0.1    -0.9    0
        # layer3    0       0       -0.1    0       0       0.8
        #   -       0       0       -0.9    0       0       -0.4
        # layer4    0       0       0       0.8     -0.4    0

        ### 1.1 get the total neural numbers ###
        for name in layer_info_dict:
            info = layer_info_dict[name]    
            print(info['weights'].shape)
            weights.append(info['weights'])

            if info['name'] == 'fc':
                # calculate fully-connected layer
                node_number = [info['out'],1,1] 
                node_layer.append(node_number)
                total_node_number += info['out']
                # print("FC",node_number, info['out'])
            elif info['name'] == 'conv2d':
                # calculate convolutional layer
                stride = info['stride']
                padding = info['padding']
                kernel_size = info['ks']
                node = []
                for i, n in enumerate(node_number):
                    if i == 2:
                        node.append(info['out'])
                        continue
                    
                    n_conv = n - (kernel_size[i]-1) + (padding[i]*2)
                    n_conv = int(n_conv/stride[i])
                    node.append(n_conv)

                node_number = [n for n in node]
                print("conv node_number", node)
                node_layer.append(node_number)
                print("====", node_layer)         
                total_node_number += node_number[0]*node_number[1]*node_number[2]

                ##### Note: We assume maxpooling here with pool size = 2 By default #####
                # Caution: do not change node_number[0], node_number[1]! We need a new array for node_number. Otherwise it will cause pointer problem
                node_number  = [node_number[0] >> 1, node_number[1] >> 1, node_number[2]]
                node_layer.append(node_number)         
                print("----", node_layer)
                total_node_number += node_number[0]*node_number[1]*node_number[2]

        A = np.zeros((total_node_number, total_node_number))
        
        ### 1.2 Calcualte the mean and std of the whole weights ###
        ## (1) calculate the mean and count of the weights
        weights = update_weights_to_robust_scale(weights,0.95)

        ## (4) update the original weight
        li = 0
        for name in layer_info_dict:
            info = layer_info_dict[name]
            info['weights'] = weights[li]
            li += 1

        print_weight_hist(weights) 
        #### 2.fill in the data (weight) for each entry ####
        layer_index = 0
        node_ptr = 0

        for name in layer_info_dict:
            # record the last layer length, current layer length with layer_index
            last_layer = node_layer[layer_index]
            last_layer_num = last_layer[0]*last_layer[1]*last_layer[2]
            layer_index += 1
            curr_layer = node_layer[layer_index]
            curr_layer_num = curr_layer[0]*curr_layer[1]*curr_layer[2]
            subA = np.zeros((last_layer_num,curr_layer_num))
            print("subA shape", subA.shape)
            info = layer_info_dict[name]
            if info['name'] == 'fc':    # simply copy the weights
                # for fully-connected layer
                subA = np.transpose(info['weights'])
                print("subA shape", info['weights'].shape)
                print("FC layer completed!")
            elif info['name'] == 'conv2d':  # much more complex
                # 1. for convolutional layer
                # Note: 1 point to n by n window nodes (we should neglect the padding)
                stride = info['stride']
                padding = info['padding']
                kernel_size = info['ks']
                weights = info['weights']
                print("weight shape", weights.shape)
                kernel_padding = [(k-1)>>1 for k in kernel_size]
                
                subA_row = np.zeros((curr_layer[0], curr_layer[1], last_layer[2]))
                for i in range(0, curr_layer[0], stride[0]):
                    for j in range(0, curr_layer[1], stride[1]):
                        for k in range(0, curr_layer[2]):
                            weight_window = weights[k]
                            last_weight = last_layer[2]
                            # TODO: Should be completed with stride problem
                            idx_i = 0
                            for p in range(i-padding[0],i+kernel_size[0]-padding[0]):
                                idx_j = 0
                                for q in range(j-padding[1],j+kernel_size[1]-padding[1]):
                                    for r in range(0, last_layer[2]):
                                        weight_subwindow = weight_window[r]
                                        if p < 0 or q < 0 or p >= last_layer[0] or q >= last_layer[1]:
                                            continue
                                        subA[p*last_layer[1]*last_layer[2]+q*last_layer[2]+r,i*curr_layer[1]*curr_layer[2]+j*curr_layer[2]+k] = weight_subwindow[idx_i,idx_j]
                                        # print(weight_subwindow[idx_i,idx_j])
                                    idx_j += 1
                                idx_i += 1
                                        
                A[node_ptr:node_ptr+last_layer_num, node_ptr+last_layer_num:node_ptr+last_layer_num+curr_layer_num] = subA
                A[node_ptr+last_layer_num:node_ptr+last_layer_num+curr_layer_num, node_ptr:node_ptr+last_layer_num] = np.transpose(subA)
                print("Conv layer completed!")
                # 2. for maxpooling layer
                node_ptr += last_layer_num    
                layer_index += 1
                curr_layer = node_layer[layer_index]
                curr_layer_num = curr_layer[0]*curr_layer[1]*curr_layer[2]    
                subA = np.zeros((last_layer_num,curr_layer_num))
                print("subA shape", subA.shape)
                for i in range(0, curr_layer[0]):
                    for j in range(0, curr_layer[1]):
                        #for k in range(0, curr_layer[2]):
                        subA[2*(i-1)*curr_layer[1]+2*(j-1),i*curr_layer[1]+j] = 1
                        subA[2*(i-1)*curr_layer[1]+2*(j-1)+1,i*curr_layer[1]+j] = 1
                        subA[(2*i-1)*curr_layer[1]+2*(j-1),i*curr_layer[1]+j] = 1
                        subA[(2*i-1)*curr_layer[1]+2*j-1,i*curr_layer[1]+j] = 1
                print("Maxpooling layer completed!")
            # update the adjacency matrix
            A[node_ptr:node_ptr+last_layer_num, node_ptr+last_layer_num:node_ptr+last_layer_num+curr_layer_num] = subA
            A[node_ptr+last_layer_num:node_ptr+last_layer_num+curr_layer_num, node_ptr:node_ptr+last_layer_num] = np.transpose(subA)
            node_ptr += last_layer_num
    with open('./A.txt', 'w') as f:
        f.write(np.array2string(A, precision=2, separator=',',max_line_width=10000))
    print("A.txt saved!")
