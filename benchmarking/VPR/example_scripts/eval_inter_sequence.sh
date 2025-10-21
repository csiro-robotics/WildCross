source ~/.bashrc
mamba activate WildCrossVPR

cd /datasets/work/d61-csirorobotics2/work/kni101/WildCross/benchmarking/VPR
export PYTHONPATH=$PWD:$PYTHONPATH

export WANDB_MODE=offline

python evaluation/eval_inter_sequence.py \
    --config configs/SALAD.yaml \
    --save_dir /scratch3/kni101/test_output \
    --pretrained_ckpt checkpoints/SALAD/split_0.pth
