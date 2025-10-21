import torch 
import pickle 
import numpy as np 
from tqdm import tqdm 
from evaluation.make_eval_sets import make_eval_sets 
from evaluation.dataloader import get_descriptors_positions

import faiss 
import pandas as pd 
import argparse 

import os 
from torchpack.utils.config import configs as CFG 
from models.model_factory import model_factory

class Evaluator:
    def __init__(self, model, env_data, split_idx, debug=False):
        self.pos_thresh = 25.0 
        self.time_thresh = 600.0
        self.recall_values = list(range(1,26))
        self.env_data = [env_data[split_idx]]
        self.model = model 
        self.image_size = CFG.val_img_size 
        self.debug = debug 
        
    def get_recalls(self, seq_data):
        name = seq_data['name']
        features = seq_data['feats']
        coords = seq_data['coords']
        timestamps = seq_data['timestamps']
    
        num_revisits = 0
        time_start = timestamps[0]
        
        feat_dists = torch.cdist(features, features)
        position_dists = torch.cdist(coords, coords)
        recalls = np.zeros(len(self.recall_values))
        
        for q_idx in tqdm(range(len(features)), desc = f"Processing Sequence {name}"):
            # Get query info 
            q_timestamp = timestamps[q_idx]
            
            # Skip if time elapsed since start is less than the time threshold
            if (q_timestamp - time_start - self.time_thresh) < 0:
                continue 
            
            # Build retrieval database 
            tt = next(x[0] for x in enumerate(timestamps) if x[1] > (q_timestamp - self.time_thresh))
            dist_seen_embedding = feat_dists[q_idx, :tt+1]
            dist_seen_world = position_dists[q_idx, :tt+1]
            
            # Check if re-visit 
            if torch.any(dist_seen_world < self.pos_thresh) and len(dist_seen_embedding) > 25:
                num_revisits += 1 
            else:
                continue 
                
            # Get distances in real world, embedding space 
            _, topk_idx = torch.topk(dist_seen_embedding, len(self.recall_values), largest=False)
            
            for i, pidx in enumerate(topk_idx):
                if dist_seen_world[pidx] <= self.pos_thresh:
                    recalls[i:] += 1 
                    break 
                
        recalls = recalls / num_revisits * 100.0 
        return recalls 

    def run(self):
        seq_data = get_descriptors_positions(self.env_data, self.image_size, self.model, self.debug)[0]
        recalls = self.get_recalls(seq_data)
        
        df_results = pd.DataFrame(columns = [f"R@{n}" for n in self.recall_values])
        df_results.loc[seq_data['name']] = recalls
        df_results = df_results.round(2)
        print(df_results)
        return df_results
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--pretrained_ckpt', type=str, required=False)
    parser.add_argument('--environments', type=str, nargs='+', default=['venman', 'karawatha'], choices=['venman','karawatha'])
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--debug', action='store_true', default=False)
    
    args, opts = parser.parse_known_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    CFG.load(args.config)
    CFG.update(opts)
    print(CFG)
    
    model = model_factory(args).cuda()
    model.eval()
    
    venman_eval_info_filepath = os.path.join(os.path.dirname(__file__), 'pickles', 'venman_eval_info.pickle')
    karawatha_eval_info_filepath = os.path.join(os.path.dirname(__file__), 'pickles', 'karawatha_eval_info.pickle')
    
    if not os.path.exists(venman_eval_info_filepath) or not os.path.exists(karawatha_eval_info_filepath):
        make_eval_sets()
    
    for env in args.environments:
        if env == 'venman':
            env_data = pickle.load(open(venman_eval_info_filepath, 'rb'))
        elif env == 'karawatha':
            env_data = pickle.load(open(karawatha_eval_info_filepath, 'rb'))
                    
    
        df_results = Evaluator(model, env_data, CFG.split_idx, args.debug).run()
        
        if args.save_dir is not None:
            save_path = os.path.join(args.save_dir, f"intrasequence_results_{env}_split_{CFG.split_idx}.csv")
            df_results.to_csv(save_path)
    