import os 
import pickle 
import numpy as np 
import pandas as pd 
from glob import glob 
from torchpack.utils.config import configs as CFG 

class MakeEvalsets:
    def __init__(self):
        self.root_folder = CFG.wildcross_path 
        
    def get_paths_coords(self, sequence):
        # Get images 
        image_paths = sorted(glob(os.path.join(self.root_folder, sequence, 'images_shrunk', '*.png')))
        
        # Get coords 
        df = pd.read_csv(os.path.join(self.root_folder, sequence, 'camera_poses_aligned.csv'))
        coords = df[['x','y','z']].to_numpy()
        
        return image_paths[::5], coords[::5]        
        
    def run(self):
        
        sequences_venman = ['V-01', 'V-02', 'V-03', 'V-04']
        sequences_karawatha = ['K-01', 'K-02', 'K-03', 'K-04']
        
        info_venman = []
        info_karawatha = []
        
        for idx, seqv in enumerate(sequences_venman):
            paths, coords = self.get_paths_coords(seqv)
            info_seq = {}
            info_seq['name'] = seqv
            info_seq['filenames'] = paths 
            info_seq['coords'] = coords 
            info_seq['timestamps'] = np.float64([os.path.basename(x).strip('.png') for x in paths])
            info_venman.append(info_seq)
            
        for seqk in sequences_karawatha:
            paths, coords = self.get_paths_coords(seqk)
            info_seq = {}
            info_seq['name'] = seqk
            info_seq['filenames'] = paths 
            info_seq['coords'] = coords 
            info_seq['timestamps'] = np.float64([os.path.basename(x).strip('.png') for x in paths])
            info_karawatha.append(info_seq) 
            
        venman_eval_info_filepath = os.path.join(os.path.dirname(__file__), 'pickles', 'venman_eval_info.pickle')
        karawatha_eval_info_filepath = os.path.join(os.path.dirname(__file__), 'pickles', 'karawatha_eval_info.pickle')
            
        with open(venman_eval_info_filepath, 'wb') as f:
            pickle.dump(info_venman, f)
        with open(karawatha_eval_info_filepath, 'wb') as f:
            pickle.dump(info_karawatha, f) 

def make_eval_sets():
    MakeEvalsets().run()
