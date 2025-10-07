source ~/.bashrc
mamba activate WildCrossVPR

cd /datasets/work/d61-csirorobotics2/work/kni101/WildCross/benchmarking/VPR
export PYTHONPATH=$PWD:$PYTHONPATH

export WANDB_MODE=offline

python train.py \
    --config configs/BoQ.yaml \
    --save_dir /scratch3/kni101/test_output \
    --pretrained_ckpt /datasets/work/d61-heatwave/work/kni101/Bag-of-Queries/pretrained_ckpts/boq_dinov2.pth
