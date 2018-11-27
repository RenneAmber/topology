import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torchvision.models as models


class DenseNet(models.DenseNet):
    def __init__(self):
        super(DenseNet, self).__init__()

    def forward_features(self, x):
        return []
