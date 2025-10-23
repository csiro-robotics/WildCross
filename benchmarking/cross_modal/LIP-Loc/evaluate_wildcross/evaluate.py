import os 
import pandas as pd 
import torch 
import argparse 
import numpy as np 
import importlib 
from evaluate_wildcross.dataloader import get_image_lidar_features
import faiss 
from tqdm import tqdm 

class Evaluator:
    def __init__(self, model, CFG, environment, split_idx, debug):
        self.model = model 
        self.CFG = CFG 
        self.environment = environment 
        self.split_idx = split_idx
        self.debug = debug 
        self.recall_values = [1,5]
        self.pos_thresh = 25.0
        
    def get_positives(self, q_coords, db_coords):
        cross_dist = torch.cdist(q_coords, db_coords)
        valid_queries = []
        valid_positives = []
        for idx, row in enumerate(cross_dist):
            positives = torch.nonzero(row <= self.pos_thresh).flatten().numpy()
            if len(positives) == 0:
                continue 
            valid_queries.append(idx)
            valid_positives.append(positives)
            
        return torch.tensor(valid_queries, dtype=torch.int32), valid_positives
    
    def get_recalls(self, query_info, db_info):
        q_name = query_info['name']
        q_feats = torch.tensor(query_info['feats'])
        q_coords = torch.tensor(query_info['coords'])
        db_name = db_info['name']
        db_feats = torch.tensor(db_info['feats'])
        db_coords =torch.tensor(db_info['coords'])
        
        db_faiss_index = faiss.IndexFlatL2(db_feats.shape[1])
        db_faiss_index.add(db_feats)
        
        valid_query_idx, positive_idx = self.get_positives(q_coords, db_coords)
        
        print(f"{q_name}->{db_name}: {len(q_coords) - len(valid_query_idx)} removed due to no valid positives")
        
        q_feats = torch.index_select(q_feats, 0, valid_query_idx)
        q_coords = torch.index_select(q_coords, 0, valid_query_idx)
        
        _, predictions = db_faiss_index.search(q_feats, max(self.recall_values))
        
        vis_recall_info = {}
        
        # Get recalls 
        recalls = np.zeros(len(self.recall_values))
        for q_idx, preds in enumerate(tqdm(predictions, desc = f"{q_name}->{db_name}")):
            vis_recall_info[valid_query_idx[q_idx]] = [
                preds[0].item(), preds[0].item() in positive_idx[q_idx]
            ]    
            
            for i, n in enumerate(self.recall_values):
                if np.any(np.isin(preds[:n], positive_idx[q_idx])):
                    recalls[i:] += 1 
                    break 
        
        # Divide by number of queries and multiply by 100 
        recalls = recalls / len(q_feats) * 100
        print(f"{q_name}->{db_name} : {recalls.round(2)}")
        return recalls, vis_recall_info
        
        
    def run(self):
        query_info, db_info_list = get_image_lidar_features(
            self.model, self.CFG, self.environment, self.split_idx, self.debug)

        df_results = pd.DataFrame(columns=["R@1", "R@5"])
        vis_recall_results = {}
        for db_info in db_info_list:
            (r1, r5), vis_recall_info = self.get_recalls(query_info, db_info)
            df_results.loc[db_info['name']] = [r1,r5]
            vis_recall_results[db_info['name']] = vis_recall_info
        df_results.loc['Average'] = df_results.mean(0)
        print(df_results.round(2))
        return df_results
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--expid', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--environments', type=str, default=['venman','karawatha'], nargs='+')
    parser.add_argument('--split_idx', type=int, required=True)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()
    
    CFG = importlib.import_module(f"config.{args.expid}").CFG
    model = importlib.import_module(f"models.{CFG.model}").Model(CFG)
    ckpt = torch.load(args.ckpt)
    ckpt = {k:v for k,v in ckpt.items() if "pooling" not in k}
    model.load_state_dict(ckpt)
    model = model.to('cuda')
    model.eval()
    print(f"Loaded pretrained model from {args.ckpt}")
    
    for env in args.environments:
        evaluator = Evaluator(model, CFG, env, args.split_idx, args.debug )
        df_results = evaluator.run()
        
        if args.save_dir is not None:
            save_path = os.path.join(args.save_dir, f"crossmodal_results_{env}_split_{args.split_idx}.csv")
            df_results.to_csv(save_path)
    