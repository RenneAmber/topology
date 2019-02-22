'''
Utilities  for saving data to disk
'''

import os
import numpy as np
import torch
import pickle as pkl 
import h5py 



def save_checkpoint(checkpoint, path, fname):
    """ Save checkpoint to path with fname """

    print('Saving checkpoint...')
    
    if not os.path.isdir(path):
        os.mkdir(path)
        
    torch.save(checkpoint, path+fname)

        
def save_activations(activs, path, fname, internal_path):
    """
    Save features to path/fname in h5py dataset.
    Features saved at <internal_path>/activations/layer_<layer>/.
    """
    
    print('Saving features...')

    ''' Create file '''
    if not os.path.exists(path):
        os.makedirs(path)
    file = h5py.File(path+fname, 'a')

    
    ''' Save features; if exists replace '''
    for i, x in enumerate(activs):
        dts = internal_path+"/activations/layer_"+str(i)
        
        if dts in file:
            data = file[dts]
            data[...] = x
        else:
            file.create_dataset(internal_path+"/activations/layer_"+str(i), data=x, dtype=np.float16)

    file.close()

    
def save_losses(losses, path, fname):
    """ Save losses (np.ndarray) to path with fname """
    if not os.path.exists(path):
        os.makedirs(path)
    
    with open(path+fname, 'wb') as f:
        pkl.dump(losses, f, protocol=2)

def save_weights(weights, path, fname):
    """ Save weights (np.ndarray of [mean, stardard deviation, min, max]) to path with fname """
    if not os.path.exists(path):
        os.makedirs(path)
    
    with open(path+fname, 'wb') as f:
        pkl.dump(weights, f, protocol=2)

def save_badj(thresholds, adj, save_dir, epoch, trial):
    from graph import binarize
    for threshold in thresholds:
        badj = binarize(np.copy(adj), threshold)
        print('t={} s={}'.format(threshold, np.sum(badj)))
        np.savetxt(save_dir + 'badj_epc%d_t%.2f_trl%d.csv' % (int(epoch), float(threshold), int(trial)), badj, fmt='%d', delimiter=",")

