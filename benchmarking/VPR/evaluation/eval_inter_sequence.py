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
        self.recall_values = list(range(1,26))
        self.env_data = env_data
        self.split_idx = split_idx 
        self.model = model 
        self.image_size = CFG.val_img_size 
        self.debug = debug 
        
    def get_positives(self, q_coords, db_coords):
        ''' 
        Given the positions of each image in the query and database sets, return 
        the ids of the positive matches for each query 
        '''
        cross_dist = torch.cdist(q_coords, db_coords)
        valid_queries = []
        valid_positives = []
        for idx, row in enumerate(cross_dist):
            positives = torch.nonzero(row <= self.pos_thresh).flatten().numpy()
            if len(positives) == 0:
                continue 
            valid_queries.append(idx)
            valid_positives.append(positives)
            
        return valid_queries, valid_positives
    
    def get_recalls(self, query_data, db_data):
        q_name = query_data['name']
        q_feats = query_data['feats']
        q_coords = query_data['coords']
        db_name = db_data['name']
        db_feats = db_data['feats']
        db_coords = db_data['coords']
        
        db_faiss_index = faiss.IndexFlatL2(db_feats.shape[1])
        db_faiss_index.add(db_feats)
        
        q_valid_idx, q_positive_idx = self.get_positives(q_coords, db_coords)
        print(f"{q_name}->{db_name}: {len(q_coords) - len(q_valid_idx)} queries removed due to no valid positives")
        
        # Remove queries, coords for examples with no valid positives before getting predictions
        q_feats = torch.index_select(q_feats, 0, torch.tensor(q_valid_idx))
        q_coords = torch.index_select(q_coords, 0, torch.tensor(q_valid_idx))
        _, predictions = db_faiss_index.search(q_feats, max(self.recall_values))
        
        # Get recalls 
        recalls = np.zeros(len(self.recall_values))
        for q_idx, preds in enumerate(tqdm(predictions, desc = f"{q_name}->{db_name}")):
            q_positives = q_positive_idx[q_idx]
            for i, n in enumerate(self.recall_values):
                if np.any(np.isin(preds[:n], q_positives)):
                    recalls[i:] += 1 
                    break 
                    
        # Divide by number of queries and multiply by 100 
        recalls = recalls / len(q_feats) * 100
        return recalls

    def run(self):
        all_sequences_data = get_descriptors_positions(self.env_data, self.image_size, self.model, self.debug)
        query_data = all_sequences_data.pop(self.split_idx)
        df_results = pd.DataFrame(columns = [f"R@{n}" for n in self.recall_values])
        
        for db_data in all_sequences_data:
            recalls = self.get_recalls(query_data, db_data)
            df_results.loc[db_data['name']] = recalls 
            
        df_results = df_results.round(2)
        print(df_results)
        return df_results
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--pretrained_ckpt', type=str, required=True)
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
            save_path = os.path.join(args.save_dir, f"intersequence_results_{env}_split_{CFG.split_idx}.csv")
            df_results.to_csv(save_path)
    