source ~/.bashrc
mamba activate WildCrossVPR

cd /path/to/WildCross/benchmarking/VPR
export PYTHONPATH=$PWD:$PYTHONPATH

export WANDB_MODE=offline

python evaluation/eval_intra_sequence.py \
    --config configs/MixVPR.yaml \
    --save_dir eval_outputs \
    --pretrained_ckpt checkpoints/MixVPR/split_1.pth \
    split_idx 1 
