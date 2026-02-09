source ~/.bashrc
mamba activate WildCrossVPR

cd /path/to/WildCross/benchmarking/VPR
export PYTHONPATH=$PWD:$PYTHONPATH

export WANDB_MODE=offline

python evaluation/eval_inter_sequence.py \
    --config configs/SALAD.yaml \
    --save_dir eval_outputs \
    --pretrained_ckpt checkpoints/SALAD/split_1.pth \
    split_idx 1 
