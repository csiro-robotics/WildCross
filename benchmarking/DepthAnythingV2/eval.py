import torch 
import argparse 

from dataset.wildcross import WildCross
from tqdm import tqdm 
from torch.utils.data import DataLoader
from depth_anything_v2.dpt import DepthAnythingV2
import torch.nn.functional as F
from util.metric import eval_depth
import torch.backends.cudnn as cudnn

cudnn.enabled = True
cudnn.benchmark = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--img_size', default=518, type=int)
    parser.add_argument('--max_depth', type=float, default=60.0)
    parser.add_argument('--max_depth_gt', type=float, default=60.0)
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()
    
    size = (args.img_size, args.img_size)
    wildcross_root = '/datasets/work/d61-csirorobotics2/source/WildCross_release'
    valset = WildCross(wildcross_root, "eval", size=size)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=16, drop_last=True, shuffle=False)
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    
    # Load pretrained checkpoint
    ckpt = torch.load(args.checkpoint)
    if 'model' in ckpt.keys():
        ckpt = ckpt['model']
    ckpt = {k.replace('module.', ''):v for k,v in ckpt.items()}
    model.load_state_dict(ckpt)
    model = model.cuda()
    
    # Evaluate!
    results = {'d1': torch.tensor([0.0]).cuda(), 'd2': torch.tensor([0.0]).cuda(), 'd3': torch.tensor([0.0]).cuda(), 
                   'abs_rel': torch.tensor([0.0]).cuda(), 'sq_rel': torch.tensor([0.0]).cuda(), 'rmse': torch.tensor([0.0]).cuda(), 
                   'rmse_log': torch.tensor([0.0]).cuda(), 'log10': torch.tensor([0.0]).cuda(), 'silog': torch.tensor([0.0]).cuda()}
    nsamples = torch.tensor([0.0]).cuda()
    model.eval()
    
    with torch.no_grad():
        for i, sample in enumerate(tqdm(valloader, desc = "Validation", total = len(valloader))):
            
            img, depth, valid_mask = sample['image'].cuda().float(), sample['depth'].cuda()[0], sample['valid_mask'].cuda()[0]
            
            with torch.no_grad():
                pred = model(img)
                pred = F.interpolate(pred[:, None], depth.shape[-2:], mode='bilinear', align_corners=True)[0, 0]
                pred = torch.clip(pred, max=args.max_depth_gt)
            
            valid_mask = (valid_mask == 1) & (depth >= 0.001) & (depth <= args.max_depth_gt)
            
            if valid_mask.sum() < 10:
                continue
            
            cur_results = eval_depth(pred[valid_mask], depth[valid_mask])
            
            for k in results.keys():
                results[k] += cur_results[k]
            nsamples += 1
            
    print('==========================================================================================')
    print('{:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}'.format(*tuple(results.keys())))
    print('{:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}'.format(*tuple([(v / nsamples).item() for v in results.values()])))
    print('==========================================================================================')
    print()
            
    
    