import argparse
import itertools

import numpy as np
import torch
from tqdm.autonotebook import tqdm
import importlib
import wandb
import os

### Global Stuff ####
parser = argparse.ArgumentParser()
parser.add_argument('--expid', type=str, required=True)
parser.add_argument('--save_dir', type=str, required=True)
parser.add_argument('--query_idx', type=int, default=None)
parser.add_argument('--pretrained', type=str, default=None)
args = parser.parse_args()

CFG = importlib.import_module(f"config.{args.expid}").CFG
CFG.expdir = args.save_dir
CFG.best_model_path = os.path.join(args.save_dir, "best.pth")
CFG.final_model_path = os.path.join(args.save_dir, "final.pth")
CFG.logdir = os.path.join(args.save_dir, "log")
CFG.query_idx = args.query_idx


model = importlib.import_module(f"models.{CFG.model}").Model(CFG)
if args.pretrained:
    ckpt = torch.load(args.pretrained)
    model.load_state_dict(ckpt)
    print(f"Loaded pretrained model from {args.pretrained}")
    
get_dataloader = importlib.import_module(f"dataloaders.{CFG.dataloader}").get_dataloader

wandb.init(project="cross-modal-place-recognition")


class AvgMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    past_filter_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items()}
        loss, num_past_filter = model(batch)
        
        past_filter_meter.update(num_past_filter, batch['camera_image'].size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        count = batch["camera_image"].size(0)
        loss_meter.update(loss.item(), count)
        
        wandb.log({"Step/loss": loss_meter.avg, "Step/percent_past_filter": past_filter_meter.avg })

        tqdm_object.set_postfix(
            train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter


def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items()}
        loss = model(batch)
        if isinstance(loss, tuple):
            loss, _ = loss 

        count = batch["camera_image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter


def main():
    dirs_to_create = [CFG.expdir, CFG.logdir]
    for dirs in dirs_to_create:
        os.makedirs(dirs, exist_ok=True)


    train_loader = get_dataloader(mode="train", CFG=CFG)
    valid_loader = get_dataloader(mode="valid", CFG=CFG)
    
    print(f"Train: {len(train_loader.dataset)}, Val: {len(valid_loader.dataset)}")

    model.to(CFG.device)
    params = [
        {
            "params": [p for p in model.encoder_camera.parameters() if p.requires_grad], 
            "lr": CFG.text_encoder_lr
        },
        {
            "params":[p for p in  model.encoder_lidar.parameters() if p.requires_grad],
            "lr": CFG.image_encoder_lr
        },
        {
            "params": [p for p in itertools.chain(model.projection_lidar.parameters(), model.projection_camera.parameters()) if p.requires_grad], 
            "lr": CFG.head_lr, 
            "weight_decay": CFG.weight_decay
        }
    ]

    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    )
    step = "epoch"

    wandb.watch(model)
    best_loss = float('inf')
    for epoch in range(CFG.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(
            model, train_loader, optimizer, lr_scheduler, step)
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader)

        wandb.log({"Epoch/train_loss": train_loss.avg, "Epoch/valid_loss": valid_loss.avg})

        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), CFG.best_model_path)
            print("Saved Best Model!")
        torch.save(model.state_dict(), CFG.final_model_path)
        lr_scheduler.step(valid_loss.avg)

    wandb.finish()


if __name__ == "__main__":
    main()
