source ~/.bashrc
conda activate LipLoc 

WILDCROSS_CODE_ROOT=/path/to/WildCross
SAVE_DIR=/path/to/save_dir
SPLIT_IDX=0 # Sequence to be held out as queries for evaluation in the cross-fold setup

cd $WILDCROSS_CODE_ROOT/benchmarking/cross_modal/LIP-Loc
export PYTHONPATH="${PYTHONPATH}:${PWD}"


python trainer_wildcross.py \
    --expid exp_wildcross_range_dinov3 \
    --save_dir $SAVE_DIR \
    --query_idx $SPLIT_IDX
