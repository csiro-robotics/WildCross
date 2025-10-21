import torch 
from models.BoQ.model import BoQModel
from models.MixVPR.model import MixVPRModel
from models.SALAD.model import SALADModel
from models.NetVLAD.model import NetVladModel
from torchpack.utils.config import configs as CFG 

def model_factory(args):
    if CFG.model_name == 'BoQ':
        model = BoQModel()
    elif CFG.model_name == 'MixVPR':
        model = MixVPRModel()
    elif CFG.model_name == 'SALAD':
        model = SALADModel()
    elif CFG.model_name == 'NetVLAD':
        model = NetVladModel()
    else:
        raise NotImplementedError(f"No method {CFG.model_name} implemented")
    
    # Load pretrained model
    if args.pretrained_ckpt is not None:
        ckpt = torch.load(args.pretrained_ckpt)
        if "state_dict" in ckpt.keys():
            ckpt = ckpt["state_dict"]
        if CFG.model_name == 'SALAD':
            if "backbone.model.pos_emb" in ckpt.keys():
                ckpt["backbone.model.pos_embed"] = ckpt.pop("backbone.model.pos_emb")
        if CFG.model_name == 'NetVLAD':
            ckpt = {k.replace('model.', 'backbone.'):v for k,v in ckpt.items()}
        model.load_state_dict(ckpt)
        print(f"Loaded pretrained model from {args.pretrained_ckpt}")
        
    return model 
