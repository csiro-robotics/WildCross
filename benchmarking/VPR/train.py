import os 
import argparse 
import torch 
import wandb
import random 
import numpy as np 
from torchpack.utils.config import configs as CFG 

from lightning.pytorch import callbacks
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger

from data.datamodule import WildCrossDataModule
from models.model_factory import model_factory


def train(save_dir):
    seed_everything(CFG.seed)
    
    model = model_factory()
    datamodule = WildCrossDataModule()
    
    wandb_logger = WandbLogger(save_dir = args.save_dir)
    wandb_logger.log_hyperparams(CFG.dict())
    
    # Define the checkpointing callback
    checkpointing = callbacks.ModelCheckpoint(
        dirpath=save_dir,
        filename='epoch-{epoch:02d}',
        every_n_epochs=10,
        save_top_k=-1,
        verbose=True,
        save_on_train_epoch_end = True,
        save_last=True,
    )
    
    # Define the progress bar callback
    program_bar = callbacks.RichProgressBar()
    
    # Define the trainer 
    trainer = Trainer(
        accelerator="gpu",
        devices=list(range(CFG.num_gpus)),
        logger=wandb_logger,          
        precision="16-mixed",
        callbacks=[checkpointing, program_bar],
        max_epochs=CFG.max_epochs,
        log_every_n_steps=1,
        fast_dev_run=False,
        default_root_dir = save_dir,
    )
    trainer.fit(model=model, datamodule=datamodule)
    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    args, opts = parser.parse_known_args()
    
    CFG.load(args.config)
    CFG.update(opts)
    print(CFG)
    
    
    

