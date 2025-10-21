import torch 
import torch.nn as nn 

from .backbone import ResNet 
from .aggregator import MixVPR
from models.lightning_model import LightningModel

from torchpack.utils.config import configs as CFG 

class MixVPRModel(LightningModel):
    def __init__(self):
        super().__init__()
        # Make model backbone             
        if "resnet" in CFG.backbone.name:
            backbone = ResNet(model_name=CFG.backbone.name, 
                              pretrained = True, 
                              layers_to_freeze = CFG.backbone.layers_to_freeze,
                              layers_to_crop=CFG.backbone.layers_to_crop)
        else:
            raise ValueError(f"backbone {CFG.backbone.name} not recognized or not implemented!")
        
        # Make model aggregator 
        aggregator = MixVPR(
            in_channels=CFG.aggregator.in_channels,
            in_h=CFG.aggregator.in_h,
            in_w=CFG.aggregator.in_w,
            out_channels=CFG.aggregator.out_channels,
            mix_depth=CFG.aggregator.mix_depth,
            mlp_ratio=CFG.aggregator.mlp_ratio,
            out_rows=CFG.aggregator.out_rows,
        )
        
        self.backbone = backbone 
        self.aggregator = aggregator
        