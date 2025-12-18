
import os 
import pickle 
import random 

import torch 
import numpy as np 
import PIL.Image as Image 

import pandas as pd 
from tqdm import tqdm 
from sklearn.neighbors import KDTree


class WildCrossInfoConstructor:
    def __init__(self,
                 data_root, 
                 img_per_place,
                 pos_include_radius,
                 pos_exclude_radius,
                 fov_include_radius,
                 anchor_framerate_downsample,
                 split_idx):
        self.dataset_path = data_root 
        self.img_per_place = img_per_place
        self.image_folder_name = 'images_shrunk'
        self.pos_include_radius = pos_include_radius 
        self.pos_exclude_radius = pos_exclude_radius 
        self.fov_include_radius = fov_include_radius 
        self.anchor_framerate_downsample = anchor_framerate_downsample
        self.split_idx = split_idx
        
    def get_positives_for_anchor(self, anchor_idx, tree, all_images, all_xyz, all_uvectors):
        # Check to see if a frame can see a corresponding place 
        anchor_xyz = all_xyz[anchor_idx]
        anchor_uvector = all_uvectors[anchor_idx]

        # Get within include radius but outside of exclude radius 
        pos_idx = tree.query_radius(np.array([anchor_xyz]), self.pos_include_radius)[0]
        exclude = tree.query_radius(np.array([anchor_xyz]), self.pos_exclude_radius)[0]
        pos_idx = [x for x in pos_idx if x not in exclude]

        # Get images, xyz, uvectors 
        pos_images = [all_images[idx] for idx in pos_idx]
        pos_xyz = np.array([all_xyz[idx] for idx in pos_idx])
        pos_uvectors = np.array([all_uvectors[idx] for idx in pos_idx])

        angle = np.arccos(np.clip(np.einsum('HD,HD->H', anchor_uvector.reshape(1,3), pos_uvectors), -1.0, 1.0)) * 180 / np.pi 
        keep_idx = np.nonzero(angle <= self.fov_include_radius)[0]

        filtered_pos_images = [pos_images[x] for x in keep_idx]
        filtered_pos_xyz = [pos_xyz[x] for x in keep_idx] 


        if len(filtered_pos_images) < self.img_per_place:
            return None 
        else:
            return {'images': filtered_pos_images, 'coords': filtered_pos_xyz}
        
    def construct_train_set_wildcross_environment(self, environment):
        if environment == 'venman':
            sequences = ['V-01', 'V-02', 'V-03', 'V-04']
        elif environment == 'karawatha':
            sequences = ['K-01', 'K-02', 'K-03', 'K-04']
        # NOTE split idx should follow convention of datasets 
        # (e.g. split_idx 1 refers to K-01 and V-01)
        sequences.pop(self.split_idx-1)
        print(f"Training Sequences for environment {environment}: {sequences}")
        
        # Get timestamp, positions and unit vectors for all images
        # We will use this to determine if an image belongs to a place 
        filepaths = []
        positions = []
        unit_vectors = []
        for seq in tqdm(sequences, desc = "Reading Sequences"):
            df = pd.read_csv(os.path.join(self.dataset_path, seq, 'camera_poses.csv'),
                             dtype = {'%time': str, 'x': np.float64, 'y': np.float64, 'z': np.float64,
                                      'qx': np.float64, 'qy': np.float64, 'qz': np.float64, 'qw': np.float64})
            
            if environment == 'karawatha':
                df['y'] += 10000 # Offset to spatially separate the two environments 

            filepaths_seq = [os.path.join(self.dataset_path, seq, self.image_folder_name, f"{ts}.png") for ts in df['%time']]
            positions_seq = df[['x','y','z']].to_numpy()
            unit_vectors_seq = (positions_seq[1:] - positions_seq[:-1]) / np.linalg.norm(positions_seq[1:] - positions_seq[:-1], axis=1, keepdims=True)
            unit_vectors_seq = np.concatenate([unit_vectors_seq, unit_vectors_seq[-1].reshape(1,-1)], axis = 0 )

            filepaths += filepaths_seq
            positions += list(positions_seq)
            unit_vectors += list(unit_vectors_seq)
        
        # Construct training set 
        anchor_idx = list(range(len(filepaths)))[::self.anchor_framerate_downsample]
        tree = KDTree(np.array(positions))
        info = []
        for aidx in tqdm(anchor_idx, desc = "Getting positives for each training idx"):
            info_idx = self.get_positives_for_anchor(aidx, tree, filepaths, positions, unit_vectors)
            if info_idx is not None:
                # print(type(info))
                info.append(info_idx)
        print(f"Total of {len(info)} Training Samples available ({len(anchor_idx) - len(info)} Excluded due to not having enough potential positives)")
        return info 
        
    def run(self):
        info_venman = self.construct_train_set_wildcross_environment('venman')
        info_karawatha = self.construct_train_set_wildcross_environment('karawatha')
        
        info_combined = info_venman + info_karawatha
        
        print(f"Total of {len(info_combined)} Training Samples ")
        print(f"Average of {np.mean([len(x['images']) for x in info_combined])} positives per training sample")
        return info_combined

class WildCrossDataset:
    def __init__(self,
                data_root,
                pos_exclude_radius,
                pos_include_radius,
                fov_include_radius,
                anchor_framerate_downsample,
                img_per_place,
                transform,
                split_idx):
        
        self.data_root = data_root
        self.pos_exclude_radius = pos_exclude_radius
        self.pos_include_radius = pos_include_radius
        self.pos_include_radius = pos_include_radius
        self.fov_include_radius = fov_include_radius
        self.anchor_framerate_downsample = anchor_framerate_downsample
        self.img_per_place = img_per_place
        self.img_per_place = img_per_place
        self.transform = transform
        self.split_idx = split_idx
        
        self.info = self.get_info()
        
    def get_info(self):
        info_pickle_path = os.path.join(os.path.dirname(__file__), 'pickles', f"train_info_pir{self.pos_include_radius}_per_{self.pos_exclude_radius}_fovr_{self.fov_include_radius}_afds_{self.anchor_framerate_downsample}_sidx_{self.split_idx}.pickle")
        if os.path.exists(info_pickle_path):
            info = pickle.load(open(info_pickle_path, 'rb'))
        else:
            os.makedirs(os.path.dirname(info_pickle_path), exist_ok=True)
            info = WildCrossInfoConstructor(
                self.data_root, self.img_per_place,
                pos_include_radius=self.pos_include_radius,
                pos_exclude_radius=self.pos_exclude_radius,
                fov_include_radius=self.fov_include_radius,
                anchor_framerate_downsample=self.anchor_framerate_downsample,
                split_idx=self.split_idx).run()
            with open(info_pickle_path, 'wb') as f:
                pickle.dump(info, f)
        return info 
    
    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        info_idx = self.info[idx]
        pos_image_paths = info_idx['images']
        pos_coords = info_idx['coords']
        
        select_idx = random.sample(list(range(len(pos_image_paths))), self.img_per_place)
        select_images = []
        select_coords = []
        
        for s_idx in select_idx:
            im = Image.open(os.path.join(self.data_root, 'images', pos_image_paths[s_idx]))
            im = self.transform(im)
            select_images.append(im)
            select_coords.append(pos_coords[s_idx])
            
        select_images = torch.tensor(np.stack(select_images))
        select_coords = torch.tensor(np.stack(select_coords))
        
        return select_images, torch.tensor(idx).repeat(self.img_per_place), select_coords
