import torch 
import torch.nn as nn 

from .NetVLAD import NetVLAD
from models.lightning_model import LightningModel

from torchpack.utils.config import configs as CFG 

class DummyAggregator(nn.Module):
    '''
    Dummy aggregator so that we can use the LightningModel template, which asks for both 
    backbone and aggregator, for NetVlad
    '''
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x 

class NetVladModel(LightningModel):
    def __init__(self):
        super().__init__()
        self.backbone = NetVLAD(descriptors_dimension=CFG.descriptor_dimension)
        self.aggregator = DummyAggregator()
