import torch 
from pytorch_metric_learning import miners
from pytorch_metric_learning.utils import common_functions as c_f

class WildVPRMultiSimilarityMiner(miners.MultiSimilarityMiner):
    def __init__(self, epsilon=0.1, **kwargs):
        super().__init__(epsilon, **kwargs)
        self.negative_thresh = 50.0

    def forward(self, embeddings, labels, positions):
        mat = self.distance(embeddings, embeddings)
        label_matches = labels.unsqueeze(1) == labels.unsqueeze(0)
        
        # Get indices of non positives 
        a2, n = torch.where(~label_matches)
        
        # Get indices of non negatives 
        dists = torch.cdist(positions, positions)
        diffs = dists > self.negative_thresh
        a1, p = torch.where(~diffs)
        
        # a1, p, a2, n = self.get_all_pairs_indices(labels, positions)
        

        if len(a1) == 0 or len(a2) == 0:
            empty = torch.tensor([], device=labels.device, dtype=torch.long)
            return empty.clone(), empty.clone(), empty.clone(), empty.clone()

        mat_neg_sorting = mat
        mat_pos_sorting = mat.clone()

        #############################################################################
        
        dtype = mat.dtype
        pos_ignore = (
            c_f.pos_inf(dtype) if self.distance.is_inverted else c_f.neg_inf(dtype)
        )
        neg_ignore = (
            c_f.neg_inf(dtype) if self.distance.is_inverted else c_f.pos_inf(dtype)
        )

        mat_pos_sorting[a2, n] = pos_ignore
        mat_neg_sorting[a1, p] = neg_ignore
        
        
        if True: # Usualy if embeddings == ref_embeddings, but always true for our setup 
            mat_pos_sorting.fill_diagonal_(pos_ignore)
            mat_neg_sorting.fill_diagonal_(neg_ignore)
            

        pos_sorted, pos_sorted_idx = torch.sort(mat_pos_sorting, dim=1)
        neg_sorted, neg_sorted_idx = torch.sort(mat_neg_sorting, dim=1)

        if self.distance.is_inverted:
            hard_pos_idx = torch.where(
                pos_sorted - self.epsilon < neg_sorted[:, -1].unsqueeze(1)
            )
            hard_neg_idx = torch.where(
                neg_sorted + self.epsilon > pos_sorted[:, 0].unsqueeze(1)
            )
        else:
            hard_pos_idx = torch.where(
                pos_sorted + self.epsilon > neg_sorted[:, 0].unsqueeze(1)
            )
            hard_neg_idx = torch.where(
                neg_sorted - self.epsilon < pos_sorted[:, -1].unsqueeze(1)
            )

        a1 = hard_pos_idx[0]
        p = pos_sorted_idx[a1, hard_pos_idx[1]]
        a2 = hard_neg_idx[0]
        n = neg_sorted_idx[a2, hard_neg_idx[1]]
        
        # Check for false positives 
        for anc, pos in zip(a1, p):
            assert labels[anc] == labels[pos], "False Positive found"
        # Check for false negatives 
        for anc, neg in zip(a2, n):
            assert labels[anc] != labels[neg], "False Negative found"

        return a1, p, a2, n

def get_max_per_row(mat, mask):
    non_zero_rows = torch.any(mask, dim=1)
    mat_masked = mat.clone()
    mat_masked[~mask] = 0
    return torch.max(mat_masked, dim=1), non_zero_rows


def get_min_per_row(mat, mask):
    non_inf_rows = torch.any(mask, dim=1)
    mat_masked = mat.clone()
    mat_masked[~mask] = float('inf')
    return torch.min(mat_masked, dim=1), non_inf_rows

class WildVPRHardTripletMiner:
    def __init__(self, margin=0.2):
        self.margin = margin 
        self.negative_thresh = 50.0
    
    def __call__(self, embeddings, labels, positions):
        embed_dist = torch.cdist(embeddings, embeddings)
        world_dist = torch.cdist(positions, positions)
        
        pos_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
        pos_mask.fill_diagonal_(False)
        
        neg_mask = world_dist > self.negative_thresh 
        
        (hardest_positive_dist, hardest_positive_indices), a1p_keep = get_max_per_row(embed_dist, pos_mask)
        (hardest_negative_dist, hardest_negative_indices), a2n_keep = get_min_per_row(embed_dist, neg_mask)
        a_keep_idx = torch.where(a1p_keep & a2n_keep)
        a = torch.arange(embed_dist.size(0)).to(hardest_positive_indices.device)[a_keep_idx]
        p = hardest_positive_indices[a_keep_idx]
        n = hardest_negative_indices[a_keep_idx]
        
        return a, p, n
