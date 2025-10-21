import torch 
import torch.nn as nn 

from .backbones.resnet import ResNet 
from .backbones.dinov2 import DINOv2
from .aggregator import SALAD
from models.lightning_model import LightningModel

from torchpack.utils.config import configs as CFG 

class SALADModel(LightningModel):
    def __init__(self):
        super().__init__()
        # Make model backbone             
        if "dinov2" in CFG.backbone.name:
            backbone = DINOv2(model_name=CFG.backbone.name, 
                              num_trainable_blocks=CFG.backbone.unfreeze_n_blocks,
                              norm_layer=True,
                              return_token=True)
        elif "resnet" in CFG.backbone.name:
            backbone = ResNet(model_name=CFG.backbone.name, 
                              pretrained = True, 
                              layers_to_freeze = CFG.backbone.layers_to_freeze,
                              layers_to_crop=CFG.backbone.layers_to_crop)
        else:
            raise ValueError(f"backbone {CFG.backbone.name} not recognized or not implemented!")
        
        # Make model aggregator 
        aggregator = SALAD(
            num_channels=CFG.aggregator.num_channels,
            num_clusters=CFG.aggregator.num_clusters,
            cluster_dim=CFG.aggregator.cluster_dim,
            token_dim=CFG.aggregator.token_dim
        )
        
        self.backbone = backbone 
        self.aggregator = aggregator
        