source ~/.bashrc
mamba activate WildCrossVPR

cd /datasets/work/d61-csirorobotics2/work/kni101/WildCross/benchmarking/VPR
export PYTHONPATH=$PWD:$PYTHONPATH

export WANDB_MODE=offline

python evaluation/eval_inter_sequence.py \
    --config configs/BoQ.yaml \
    --save_dir /scratch3/kni101/test_output \
    --pretrained_ckpt /scratch3/kni101/work/Bag-of-Queries/batch_jobs/August/27/train_intersequence/BoQ/epoch-epoch=39.ckpt
