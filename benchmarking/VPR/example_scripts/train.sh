source ~/.bashrc
mamba activate WildCrossVPR

cd /path/to/WildCross/benchmarking/VPR
export PYTHONPATH=$PWD:$PYTHONPATH

export WANDB_MODE=offline

python train.py \
    --config configs/BoQ.yaml \
    --save_dir BoQ_output \
    --pretrained_ckpt /path/to/pretrained_ckpts/boq_dinov2.pth
