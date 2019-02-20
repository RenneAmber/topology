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

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--net')
parser.add_argument('--dataset')
parser.add_argument('--trial', default=0, type=int)
parser.add_argument('--epochs', nargs='+', type=int)
parser.add_argument('--split', type=int, default=0)
parser.add_argument('--kl', type=int, default=0)
parser.add_argument('--input_size', default=32, type=int)
parser.add_argument('--thresholds', nargs='+', type=float)
parser.add_argument('--function_type', default=0, type=int)

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
''' Meta-name to be used as prefix on all savings'''
oname = args.net + '_' + args.dataset + '/'
SAVE_PATH = '../../data/'
#SAVE_PATH = '/data/data1/datasets/cvpr2019/'
SAVE_DIR = SAVE_PATH + 'adjacency/' + oname
START_LAYER = 3 if args.net in ['vgg', 'resnet'] else 0 
THRESHOLDS = args.thresholds
FUNCTION_TYPE = args.function_type
print(THRESHOLDS)
''' If save directory doesn't exist create '''
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)    

# Build models
print('==> Building model..')
net = get_model(args.net, args.dataset)
net = net.to(device)
print(net)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

''' Prepare criterion '''
if args.dataset in ['cifar10', 'imagenet']:
    criterion = nn.CrossEntropyLoss()
elif args.dataset in ['mnist', 'mnist_adverarial']:
    criterion = F.nll_loss

# save badj csv file
def save_badj(epoch, adj):
    for threshold in THRESHOLDS:
        badj = binarize(np.copy(adj), threshold)
        print('t={} s={}'.format(threshold, np.sum(badj)))
        np.savetxt(SAVE_DIR + 'badj_epc%d_t%.3f_trl%d.csv' % (int(epoch), float(threshold), int(args.trial)), badj, fmt='%d', delimiter=",")

weight_save_dict = {}
for epoch in args.epochs:
    print('==> Loading checkpoint for epoch {}...'.format(epoch))
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/'+ args.net + '_' + args.dataset + '/ckpt_trial_' + str(args.trial) + '_epoch_' + str(epoch)+'.t7')
    net.load_state_dict(checkpoint['net'])
    
    ''' Define passer and get activations '''
    functloader = loader(args.dataset+'_test', batch_size=100, subset=list(range(0, 1000)))
    passer = Passer(net, functloader, criterion, device)
    if FUNCTION_TYPE == 1:    # Whole weights of the network
        weights, weight_save = passer.get_structure()
        ''' Save [mean, std, min, max] of the weights in every epoch '''
        weight_save_dict['epc_{}'.format(epoch)] = weight_save
        ''' If high number of nodes compute adjacency on layers and chunks'''

        ''' Treat all network at once or split it into chunks and treat each '''
        if not args.split:
            splits = signal_dimension_adjusting(weights,weights[0].shape[1])
            print("Splits number:{}".format(splits[0].shape)) 
            weights = signal_concat(splits)
            
            adj = adjacency(weights,shape=1)
            
            save_badj(epoch, adj)
        else:
            print("weights shape as : {}, {}".format(weights[0].shape,weights[0].shape[1]))
            splits = signal_splitting(weights, weights[0].shape[1])
            
            if not args.kl:
                ''' Compute correlation metric for each split'''
                adjs = [[adjacency(x) for x in layer] for layer in splits]
                for threshold in THRESHOLDS:
                    save_splits(adjs, args.split, SAVE_DIR, START_LAYER, epoch, threshold, args.trial)
            else:
                ''' Compute KL divergence between correlation distribution of each pair of splits '''
                adj = adjacency_kl(splits)
                for threshold in THREHSOLDS:
                    np.savetxt(SAVE_DIR + 'badj_epc{}_t{:1.2f}_trl{}.csv'.format(epoch, threshold, args.trial), adj, fmt='%d', delimiter=",")
    elif FUNCTION_TYPE == 2:    # Single layer of the network
        weight = passer.get_structure_layer(3)  
        weight_save_dict['epc_{}'.format(epoch)] = weight
        weights = signal_concat(weight)
        
        adj = adjacency(weights,shape=1)
        
        save_badj(epoch, adj)
    elif FUNCTION_TYPE == 3:    # Node with weights
        node_number = [0,0,0]   # the node number of the last layer (in three dimension)
        node_layer = []         # node number for each layer
        total_node_number = 0   # total node number 
        if args.dataset == 'mnist' or args.dataset == 'cifar10':
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
                node_layer.append(node_number)     
                total_node_number += node_number[0]*node_number[1]*node_number[2]

                ##### Note: We assume maxpooling here with pool size = 2 By default #####
                # Caution: do not change node_number[0], node_number[1]! We need a new array for node_number. Otherwise it will cause pointer problem
                node_number  = [node_number[0] >> 1, node_number[1] >> 1, node_number[2]]
                node_layer.append(node_number)         
                total_node_number += node_number[0]*node_number[1]*node_number[2]

        print(node_layer)  
        A = np.zeros((total_node_number, total_node_number))
        
        ### 1.2 (1)(2)(3) Calcualte the mean and std of the whole weights ###
        # Approach 1. Gaussian distribution
        # weights = update_weights_to_distribution(weights)

        # Approach 2. Robust scale
        weights = update_weights_to_robust_scale(weights,0.95)    
        
        ## (4) update the original weight
        li = 0
        for name in layer_info_dict:
            info = layer_info_dict[name]
            info['weights'] = weights[li]
            li += 1

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
                
                ''' version 1
                print("subA shape", subA.shape)
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
                                    idx_j += 1
                                idx_i += 1
                '''
                # version 2
                print("subA shape", subA.shape)
                for i in range(0, last_layer[0]):
                    for j in range(0, last_layer[1]):
                        for k in range(0, last_layer[2]):
                            # TODO: Should be completed with stride problem
                            idx_i = 0
                            for p in range(i-padding[0],i+kernel_size[0]-padding[0]):
                                idx_j = 0
                                for q in range(j-padding[1],j+kernel_size[1]-padding[1]):
                                    if p < 0 or q < 0 or p >= curr_layer[0] or q >= curr_layer[1]:
                                       idx_j += 1     
                                       continue
                                    for r in range(0, curr_layer[2]):
                                        weight_subwindow = weights[r][k]
                                        subA[i*last_layer[1]*last_layer[2]+j*last_layer[2]+k,p*curr_layer[1]*curr_layer[2]+q*curr_layer[2]+r] = weight_subwindow[idx_i,idx_j]
                                    idx_j += 1
                                idx_i += 1
                     
                A[node_ptr:node_ptr+last_layer_num, node_ptr+last_layer_num:node_ptr+last_layer_num+curr_layer_num] = subA
                A[node_ptr+last_layer_num:node_ptr+last_layer_num+curr_layer_num, node_ptr:node_ptr+last_layer_num] = np.transpose(subA)
                print("Conv layer completed!")
                # 2. for maxpooling layer
                node_ptr += last_layer_num   
                last_layer_num = curr_layer_num 
                layer_index += 1
                curr_layer = node_layer[layer_index]
                curr_layer_num = curr_layer[0]*curr_layer[1]*curr_layer[2]    
                subA = np.zeros((last_layer_num,curr_layer_num))
                print("subA shape", subA.shape,"maxpooling -- curr_layer", curr_layer)
                for i in range(0, curr_layer[0]):
                    for j in range(0, curr_layer[1]):
                        for k in range(0, curr_layer[2]):
                            subA[2*i*curr_layer[1]*curr_layer[2]+2*j*curr_layer[2],         i*curr_layer[1]*curr_layer[2]+j*curr_layer[2]+k] = 1
                            subA[2*i*curr_layer[1]*curr_layer[2]+(2*j+1)*curr_layer[2],     i*curr_layer[1]*curr_layer[2]+j*curr_layer[2]+k] = 1
                            subA[(2*i+1)*curr_layer[1]*curr_layer[2]+2*j*curr_layer[2],     i*curr_layer[1]*curr_layer[2]+j*curr_layer[2]+k] = 1
                            subA[(2*i+1)*curr_layer[1]*curr_layer[2]+(2*j+1)*curr_layer[2], i*curr_layer[1]*curr_layer[2]+j*curr_layer[2]+k] = 1
                # print("Maxpooling layer completed!")
            # update the adjacency matrix
            A[node_ptr:node_ptr+last_layer_num, node_ptr+last_layer_num:node_ptr+last_layer_num+curr_layer_num] = subA
            A[node_ptr+last_layer_num:node_ptr+last_layer_num+curr_layer_num, node_ptr:node_ptr+last_layer_num] = np.transpose(subA)
            node_ptr += last_layer_num

        
        
        if not args.split: 
            adj = np.abs(np.nan_to_num(A))
            print("adjacency shape is", adj.shape)
            save_badj(epoch, adj)
        else:
            splits = signal_splitting(A,1000)
            
            if not args.kl:
                ''' Compute correlation metric for each split'''
                adjs = [[adjacency(x) for x in layer] for layer in splits]
                for threshold in THRESHOLDS:
                    save_splits(adjs, args.split, SAVE_DIR, START_LAYER, epoch, threshold, args.trial)
            else:
                ''' Compute KL divergence between correlation distribution of each pair of splits '''
                adj = adjacency_kl(splits)
                for threshold in THREHSOLDS:
                    np.savetxt(SAVE_DIR + 'badj_epc%d_t%.2f_trl%d.csv' % (int(epoch), float(threshold), int(args.trial)), badj, fmt='%d', delimiter=",")

weight_path = './weights/{}_{}/'.format(args.net, args.dataset)
weight_fname = 'trial_{}.pkl'.format(args.trial)
save_weights(weight_save_dict, weight_path, weight_fname)
print("=====> Weight saved! ")
               
        
