
import os
import numpy as np
import torch

PATH = '../data/adjacency/lenet_mnist/'

files = os.listdir(PATH)
for item in files:
    if item.endswith('.txt'):
        os.remove(os.path.join(PATH,item))
        print("Remove item:",item)




