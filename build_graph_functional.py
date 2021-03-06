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
from savers import save_activations, save_checkpoint, save_losses
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
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
''' Meta-name to be used as prefix on all savings'''
oname = args.net + '_' + args.dataset + '/'
SAVE_PATH = '/data/data1/datasets/cvpr2019/'
SAVE_DIR = SAVE_PATH + 'adjacency/' + oname
START_LAYER = 3 if args.net in ['vgg', 'resnet'] else 0 
THRESHOLDS = args.thresholds

''' If save directory doesn't exist create '''
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)    

# Build models
print('==> Building model..')
net = get_model(args.net, args.dataset)
net = net.to(device)
    
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

''' Prepare criterion '''
if args.dataset in ['cifar10', 'imagenet']:
    criterion = nn.CrossEntropyLoss()
elif args.dataset in ['mnist', 'mnist_adverarial']:
    criterion = F.nll_loss


for epoch in args.epochs:
    print('==> Loading checkpoint for epoch {}...'.format(epoch))
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/'+ args.net + '_' + args.dataset + '/ckpt_trial_' + str(args.trial) + '_epoch_' + str(epoch)+'.t7')
    net.load_state_dict(checkpoint['net'])
    
    ''' Define passer and get activations '''
    functloader = loader(args.dataset+'_test', batch_size=100, subset=list(range(0, 1000)))
    passer = Passer(net, functloader, criterion, device)
    activs = passer.get_function()

    ''' If high number of nodes compute adjacency on layers and chunks'''

    ''' Treat all network at once or split it into chunks and treat each '''
    if not args.split:
        activs = signal_concat(activs)
        adj = adjacency(activs)
        
        for threshold in THRESHOLDS:
            badj = binarize(np.copy(adj), threshold)
            print('t={} s={}'.format(threshold, np.sum(badj)))
            np.savetxt(SAVE_DIR + 'badj_epc{}_t{:1.2f}_trl{}.csv'.format(epoch, threshold, args.trial), badj, fmt='%d', delimiter=",")
    else:
        splits = signal_splitting(activs, args.split)
        
        if not args.kl:
            ''' Compute correlation metric for each split'''
            adjs = [[adjacency(x) for x in layer] for layer in splits]
            for threhold in THRESHOLDS:
                save_splits(adjs, args.split, SAVE_DIR, START_LAYER, epoch, threshold, args.trial)
        else:
            ''' Compute KL divergence between correlation distribution of each pair of splits '''
            adj = adjacency_kl(splits)
            for threshold in THREHSOLDS:
                np.savetxt(SAVE_DIR + 'badj_epc{}_t{:1.2f}_trl{}.csv'.format(epoch, threshold, args.trial), adj, fmt='%d', delimiter=",")
                



                
                




        
