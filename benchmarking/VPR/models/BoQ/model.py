import torch 
import torch.nn as nn 

from .backbones import DinoV2, ResNet 
from .aggregator import BoQ
from models.lightning_model import LightningModel

from torchpack.utils.config import configs as CFG 

class BoQModel(LightningModel):
    def __init__(self):
        super().__init__()
        # Make model backbone 
        if "dinov2" in CFG.backbone.name:
            backbone = DinoV2(backbone_name=CFG.backbone.name, 
                              unfreeze_n_blocks=CFG.backbone.unfreeze_n_blocks)
            
        elif "resnet" in CFG.backbone.name:
            backbone = ResNet(backbone_name=CFG.backbone.name, 
                              unfreeze_n_blocks=CFG.backbone.unfreeze_n_blocks, 
                              crop_last_block=True)
        else:
            raise ValueError(f"backbone {CFG.backbone.name} not recognized or not implemented!")
        
        # Make model aggregator 
        aggregator = BoQ(
            in_channels=backbone.out_channels,
            proj_channels = CFG.aggregator.channel_proj,
            num_queries = CFG.aggregator.num_queries,
            num_layers = CFG.aggregator.num_layers,
            row_dim = CFG.aggregator.output_dim // CFG.aggregator.channel_proj,
        )
        
        self.backbone = backbone 
        self.aggregator = aggregator
        