import torch 
import numpy as np 
import os 
from glob import glob 
import albumentations as A
from utils.rangeimage_utils_wildcross import loadCloudFromBinary, createRangeImageWildVPR
from utils.model_utils import get_transforms
import cv2 
import pandas as pd 
from tqdm import tqdm 
import torch.nn.functional as F 
from evaluate_wildcross.dataloader_utils import transform_image_trajectory
import yaml 

class EvalDataset:
    def __init__(self, CFG, seq, mode):
        self.transforms = get_transforms('val', size=CFG.size)
        self.CFG = CFG 
        self.data_path = CFG.data_path 
        self.mode = mode 
        if mode == 'lidar':
            self.filepaths = sorted(glob(os.path.join(
                self.data_path, seq, 'Clouds', '*bin')))
        elif mode == 'image':
            self.filepaths = sorted(glob(os.path.join(
                self.data_path, seq, 'images_shrunk', '*png'
            )))[900:][::5]
        else:
            raise ValueError(mode) 
        
        
        
    def __len__(self):
        return len(self.filepaths)
    
    def __getitem__(self, idx):
        batch = {}
        if self.mode == 'lidar':
            points = loadCloudFromBinary(self.filepaths[idx])
            image = createRangeImageWildVPR(points, self.CFG.crop, self.CFG.crop_distance, self.CFG.distance_threshold)
            image = self.transforms(image=image)['image']
            image = torch.tensor(image).permute(2,0,1).float()
            batch['lidar_image'] = image 
        elif self.mode == 'image':
            image = cv2.cvtColor(cv2.imread(self.filepaths[idx]), cv2.COLOR_BGR2RGB)
            image = self.transforms(image=image)['image']
            image = torch.tensor(image).permute(2,0,1).float()
            batch['camera_image'] = image 
        return batch 

@torch.no_grad()
def get_seq_feats_coords(model, CFG, seq, mode, debug=False):
    # Get poses 
    if mode == 'image':
        xyz = pd.read_csv(os.path.join(
            CFG.data_path, seq, 'camera_poses_aligned.csv'))[['x','y','z']]
        print(len(xyz))
        xyz = xyz.to_numpy()[900:][::5]
        print(len(xyz))
        config_path = glob(os.path.join(CFG.data_path, seq, 'p*yaml'))[0]
        config = yaml.safe_load(open(config_path))
        xyz = transform_image_trajectory(xyz, config)
        
    elif mode == 'lidar':
        xyz = pd.read_csv(os.path.join(
            CFG.data_path, seq, 'submap_poses_aligned.csv'))[['x','y','z']]
        xyz = xyz.to_numpy()
    
    if debug:
        all_feats = F.normalize(torch.randn(len(xyz), 256), dim = 1)
    else:
        # Get features
        dataset = EvalDataset(CFG, seq, mode)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size = 32, 
            num_workers  = 16,
            shuffle = False,
            pin_memory=True 
        )
        
        all_feats = []
        if mode == 'image':
            desc = f"Getting Query Features (Seq. {seq})"
        elif mode == 'lidar':
            desc = f"Getting Database Features (Seq. {seq})"
            
        for batch in tqdm(loader, desc=desc):
            batch = {k:v.cuda() for k,v in batch.items()}
            if mode == 'image':
                feats = model.get_camera_embeddings(batch).cpu().numpy()
            elif mode == 'lidar':
                feats = model.get_lidar_embeddings(batch).cpu().numpy()
            all_feats.append(feats)
        
        all_feats = np.concatenate(all_feats)

    assert len(all_feats) == len(xyz), f"{seq}, {len(all_feats)}, {len(xyz)}"

    return {"name": seq, "feats": all_feats, "coords": xyz }
        
        
        
    
        
    
    
def get_image_lidar_features(model, CFG, environment, query_idx, debug):
    if environment == 'venman':
        sequences = ['V-01','V-02','V-03','V-04']
    elif environment == 'karawatha':
        sequences = ['K-01','K-02','K-03','K-04']
    query_sequence = sequences.pop(query_idx)
    query_info = get_seq_feats_coords(model, CFG, query_sequence, mode='image', debug=debug)
    db_info_list = []
    for seq in sequences:
        db_info_list.append(get_seq_feats_coords(model, CFG, seq, mode='lidar', debug=debug))
    
    return query_info, db_info_list 
    

            