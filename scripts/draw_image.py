import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt



parser = argparse.ArgumentParser(description='PyTorch draw images ')
parser.add_argument('--net')
parser.add_argument('--dataset')
parser.add_argument('--trial', default=0, type=int)
args = parser.parse_args()

''' Load bettis '''
PATH = '../../data/adjacency/{}_{}/'.format(args.net, args.dataset)
filename = 'bettis_trl{}_p0.0_s1.0.pkl'.format(args.trial)
with open(PATH+filename,"rb") as f:
    epc_dict = pickle.load(f)

''' Load loss and accuracy'''
LOSS_PATH = './losses/{}_{}/'.format(args.net, args.dataset)
loss_filename = 'stats_trial_{}.pkl'.format(args.trial)
with open(LOSS_PATH+loss_filename,"rb") as f:
    loss_dict = pickle.load(f)  

''' Load weights '''
WEIGHT_PATH = './weights/{}_{}/'.format(args.net, args.dataset)
weight_filename = 'trial_{}.pkl'.format(args.trial)
with open(WEIGHT_PATH+weight_filename, "rb") as f:
    weight_dict = pickle.load(f)

def draw_gabor_filter():
    c = [f for f in weight_dict]
    fig, axs = plt.subplots(nrows=len(c),ncols=10,figsize=(9.3, 6),
                        subplot_kw={'xticks': [], 'yticks': []})
    convs = []
    for epc in weight_dict:
        print(epc)
        for array in weight_dict[epc]:
            print("====")
            carray = np.reshape(array.astype(float), (5, 5))
            convs.append(carray)
            print(carray)
    for i,ax in enumerate(axs.flat):
        ax.imshow(convs[i],cmap='gray')
    plt.tight_layout()
    plt.show()

#draw_gabor_filter()
if(not epc_dict or not loss_dict):
    print("Error input filename:{}, please check.".format(filename))
else:
    colors = ['lightpink','violet','crimson','hotpink','magenta','m','purple','k','g','b','r','y','c']
    bettis1_all = []
    bettis1_epc0 = []
    # epc_dict['epc_4'][][(betti_number1/2/3)]
    for index, epc in enumerate(epc_dict):
        epc_num = int(epc[4:])
        epc_label = "%s(tr%.2f,te%.2f)" % (epc, loss_dict[epc_num]['acc_tr'], loss_dict[epc_num]['acc_te'])
        if epc_num == 0:
            bettis1_epc0 = [epc_dict[epc][t][0] for t in epc_dict[epc]]
            continue
        bettis1 = [(epc_dict[epc][t][0]-bettis1_epc0[i])/bettis1_epc0[i] for i,t in enumerate(epc_dict[epc])]
        bettis2 = [epc_dict[epc][t][1] for t in epc_dict[epc]]
        bettis3 = [epc_dict[epc][t][2] for t in epc_dict[epc]]
        print(epc, bettis1)
        #print("======",weight_dict[epc])
        #print(epc, bettis2)
        #print(epc, bettis3)
        edge_density = [t[2:] for t in epc_dict[epc]]
        plt.plot( edge_density[:], bettis1[:], color=colors[index % 13], label=epc_label)
        # plt.plot( edge_density[0:20], bettis2[0:20], color=colors[index % 12], label=epc_label)
        # plt.plot( edge_density[0:20], bettis3[0:20], color=colors[index % 12], label=epc_label)
    plt.legend(loc='best')
    plt.show()
    

