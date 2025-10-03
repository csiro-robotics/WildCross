from models.BoQ.model import BoQModel
from torchpack.utils.config import configs as CFG 

def model_factory():
    if CFG.model_name == 'BoQ':
        model = BoQModel()
        
    else:
        raise NotImplementedError(f"No method {CFG.model_name} implemented")
    return model 
