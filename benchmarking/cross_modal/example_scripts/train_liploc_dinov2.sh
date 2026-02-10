source ~/.bashrc
conda activate LipLoc 

WILDCROSS_CODE_ROOT=/path/to/WildCross
SAVE_DIR=/path/to/save_dir
SPLIT_IDX=1 # Sequence to be held out as queries for evaluation in the cross-fold setup

cd $WILDCROSS_CODE_ROOT/benchmarking/cross_modal/
export PYTHONPATH="${PYTHONPATH}:${PWD}"


python trainer_wildcross.py \
    --expid exp_wildcross_range_dinov2 \
    --save_dir $SAVE_DIR \
    --split_idx $SPLIT_IDX
