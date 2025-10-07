import torch 
from models.BoQ.model import BoQModel
from torchpack.utils.config import configs as CFG 

def model_factory(args):
    if CFG.model_name == 'BoQ':
        model = BoQModel()
        
    else:
        raise NotImplementedError(f"No method {CFG.model_name} implemented")
    
    # Load pretrained model
    if args.pretrained_ckpt is not None:
        ckpt = torch.load(args.pretrained_ckpt)
        if "state_dict" in ckpt.keys():
            ckpt = ckpt["state_dict"]
        if CFG.model_name == 'SALAD':
            ckpt = {k.replace('model.', ''):v for k,v in ckpt.items()}
            if "backbone.model.pos_emb" in ckpt.keys():
                ckpt["backbone.model.pos_embed"] = ckpt.pop("backbone.model.pos_emb")
        model.load_state_dict(ckpt)
        print(f"Loaded pretrained model from {args.pretrained_ckpt}")
        
    return model 
