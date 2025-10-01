source ~/.bashrc
conda activate LipLoc 

WILDCROSS_ROOT=/datasets/work/d61-csirorobotics2/work/kni101/WildCross
SAVE_DIR=/scratch3/kni101/SAVEDIR
QIDX=0 # Sequence to be held out as queries for evaluation in the cross-fold setup

cd $WILDCROSS_ROOT/benchmarking/cross_modal/LIP-Loc
export PYTHONPATH="${PYTHONPATH}:${PWD}"


python trainer_wildcross.py \
    --expid exp_wildvpr_range_dinov2 \
    --save_dir $SAVE_DIR \
    --query_idx $QIDX
