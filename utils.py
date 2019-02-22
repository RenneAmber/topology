'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.transforms as transforms
import torchvision
import numpy as np
import h5py
import errno
import os.path
import random

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

try:
    _, term_width = os.popen('stty size', 'r').read().split()
except ValueError:
    term_width = 0
    
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def save_splits(splits, split_size, save_dir, start_layer, epoch, threshold, trial):
    ''' Some description '''
    for i_layer, layer in enumerate(splits):
        for i_chunk, chunk in enumerate(layer):
            path = save_dir + '/{}/layer{}_chunk{}/'.format(split_size, start_layer + i_layer, i_chunk)
        
            if not os.path.exists(path):
                os.makedirs(path)

            print('Saving ... trl{}, epc{}, threshold{:1.2f}, layer{}, chunk{}, shape {}'.format(trial, epoch, threshold, start_layer+i_layer, i_chunk, chunk.shape))
            np.savetxt(path+'badj_epc{}_t{:1.2f}_trl{}.csv'.format(epoch, threshold, trial), chunk, fmt='%d', delimiter=",")

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
            if weights[i] < 0:
                weights[i] = -weights[i]

##### Update weights so as to satisfy some distribution
def update_weights_to_normal_distribution(weights):
    ### 1.2 Calcualte the mean and std of the whole weights ###
        
    weight_count = 0 
    weight_sum = 0
    weight_std = 0
    ## (1) calculate the mean and count of the weights
    for w in weights:
        l = len(w.shape)
        s,c = calculate_sum(w, 1, l)    
        weight_sum += s
        weight_count += c
    
    weight_mean = weight_sum/weight_count
    print(weight_mean, weight_sum, weight_count)
   
    ## (2) calculate the standard deviation through mean
    
    for w in weights:
        l = len(w.shape)
        weight_std += calculate_std(w, weight_mean, 1, l)    
        
    weight_std = math.sqrt(weight_std/weight_count)
    print("weight_std = ", weight_std)
    ## (3) transform the distribution to normal distribution
    for w in weights:
        l = len(w.shape)
        transform_distribution(w, weight_mean, weight_std, 1, l)

    return weights

def update_weights_to_robust_scale(weights, percent):
    #### Method: w' = (w - mid(W))/(Pmax - Pmin)
    # where mid(W) is the median of weights, Pmax is the weights with smaller percent%, Pmin is the weights with smaller (1-percent)%
    vweights = []
    weight_count = 0
    for w in weights:
        l = len(w.shape)
        _vectorize_weights(w,vweights,1,l)
    # (1) calculate the median
    weight_count = len(vweights)
    vweights.sort()
    print("[utils update_weights_to_robust_scale] weight count is",weight_count)
    weight_median = 0.5 * (vweights[weight_count>>1] + vweights[(weight_count>>1)+1])

    # (2) calculate Pmax and Pmin
    pmin = int((1-percent)*weight_count)
    pmax = int(percent*weight_count)    
    weight_std = vweights[pmax]-vweights[pmin]

    # (3) update weights
    for w in weights:
        l = len(w.shape)
        print("curr layer shape is", w.shape)
        transform_distribution(w, weight_median, weight_std, 1, l)

    return weights
def _vectorize_weights(weights, v, level, total_level):
    for w in weights:
        if level<total_level:
            _vectorize_weights(w,v, level+1, total_level)
        else:
            v.append(w)
#### Print weight distribution histogram ####
def print_weight_hist(weights):
    vweights = []
    for w in weights:
        l = len(w.shape)
        _vectorize_weights(w,vweights,1,l)
    
    import matplotlib.pyplot as plt
    plt.hist(vweights,600)
    plt.show()

