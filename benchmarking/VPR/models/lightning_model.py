import torch 
from pytorch_metric_learning import losses, reducers
from pytorch_metric_learning.distances import LpDistance

import lightning as L 
from torchmetrics.aggregation import RunningMean
from torchpack.utils.config import configs as CFG 

from models.miner import WildCrossMultiSimilarityMiner, WildCrossHardTripletMiner

class LightningModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.backbone = None # Overwritten by children classes
        self.aggregator = None # Overwritten by children classes 
        
        self.lr = CFG.lr 
        self.lr_mul = CFG.lr_mul 
        self.weight_decay = CFG.weight_decay 
        self.warmup_epochs = CFG.warmup_epochs 
        self.milestones = CFG.milestones 
        
        # Init loggers 
        self.running_mean_loss = RunningMean(window=500)
        self.running_mean_batch_acc = RunningMean(window=500)
        
        # Init loss function and miner 
        if CFG.loss == 'multi_similarity_loss':
            self.loss = losses.MultiSimilarityLoss(alpha=1, beta=50, base=0.)
            self.miner = WildCrossMultiSimilarityMiner(epsilon=0.1)
        elif CFG.loss == 'hard_triplet_loss':
            distance = LpDistance(normalize_embeddings=False, collect_stats=True)
            reducer_fn = reducers.AvgNonZeroReducer(collect_stats=True)
            self.loss = losses.TripletMarginLoss(margin=0.2, swap=True, distance=distance,
                                                reducer=reducer_fn, collect_stats=True)
            self.miner = WildCrossHardTripletMiner(margin=0.2)
        
    def configure_optimizers(self):
        optimizer_params = [
            {"params": self.backbone.parameters(),   "lr": self.lr, "weight_decay": self.weight_decay},
            {"params": self.aggregator.parameters(), "lr": self.lr, "weight_decay": self.weight_decay},
        ]
        optimizer = torch.optim.AdamW(optimizer_params)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.milestones, gamma=self.lr_mul
        )    
        return [optimizer], [scheduler]
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # warmup learning rate for the first `self.warmup_epochs` epochs
        if self.trainer.current_epoch < self.warmup_epochs:
            total_warmup_steps = self.warmup_epochs * self.trainer.num_training_batches
            lr_scale = (self.trainer.global_step + 1) / total_warmup_steps
            lr_scale = min(1.0, lr_scale)
            for pg in optimizer.param_groups:
                initial_lr = pg.get("initial_lr", self.lr)
                pg["lr"] = lr_scale * initial_lr
        
        optimizer.step(closure=optimizer_closure)
        self.log('_LR_aggregator', optimizer.param_groups[-1]['lr'], prog_bar=False, logger=True)
        self.log('_LR_backbone', optimizer.param_groups[-2]['lr'], prog_bar=False, logger=True)

    def compute_loss(self, descriptors, labels, positions):
        mined_pairs = self.miner(descriptors, labels, positions)
        loss = self.loss(descriptors, labels, mined_pairs)
        
        nb_samples = descriptors.shape[0]
        nb_mined = len(set(mined_pairs[0].detach().cpu().numpy()))
        batch_acc = 1.0 - (nb_mined / nb_samples)
        
        return loss, batch_acc

    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregator(x)
        if isinstance(x, tuple):
            x = x[0]
        return x 
    
    def training_step(self, batch, batch_idx):
        images, labels, positions = batch 
        images = images.flatten(0, 1)
        labels = labels.flatten()
        positions = positions.flatten(0, 1)
        
        # Forward pass 
        descriptors = self(images)
        loss, batch_acc = self.compute_loss(descriptors, labels, positions)
        self.running_mean_loss.update(loss.item())
        self.running_mean_batch_acc.update(batch_acc)
        
        self.log("loss", loss, prog_bar=True, logger=True)
        self.log("loss_running_mean", self.running_mean_loss.compute(), prog_bar=True, logger=True)
        
        self.log("batch_acc", batch_acc)
        self.log("batch_acc_running_mean", self.running_mean_batch_acc.compute())
        return loss 
        
    def on_train_epoch_end(self):
        self.log("epoch_end", 1.0)
        
    
        