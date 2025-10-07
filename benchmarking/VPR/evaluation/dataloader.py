import torchvision.transforms as T
import PIL.Image as Image 
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
import torch 
from tqdm import tqdm 

class EvalDataset:
    def __init__(self, filenames, image_size):
        self.filenames = filenames 
        transforms = [  T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        T.Resize(size=image_size, antialias=True)]
        self.transform = T.Compose(transforms)
        
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        im = Image.open(self.filenames[idx]).convert("RGB")
        im = self.transform(im)
        return im 
    
@torch.no_grad()
def get_descriptors_positions(env_data, image_size, model, debug=False):
    info = []
    for seq_data in env_data:
        seq_dataset = EvalDataset(filenames=seq_data['filenames'], image_size=image_size)
        seq_dataloader = DataLoader(seq_dataset,
                                    batch_size = 32,
                                    num_workers=16,
                                    pin_memory=True,
                                    shuffle=False)
        seq_feats = []
        if debug:
            seq_feats = F.normalize(torch.randn(len(seq_dataset), 12288), dim = -1)
        else:
            for batch in tqdm(seq_dataloader, desc = f"Extracting features for Seq. {seq_data['name']}"):
                batch = batch.to('cuda')
                feats = model.forward(batch)
                seq_feats.append(feats.cpu())
            seq_feats = torch.cat(seq_feats)
        seq_pos = torch.tensor(seq_data['coords'])
        seq_ts = torch.tensor(seq_data['timestamps'])
        info.append({'name': seq_data['name'], 'feats': seq_feats, 'coords': seq_pos, 'timestamps': seq_ts})
    return info 
