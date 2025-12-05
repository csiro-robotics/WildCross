_WILDCROSS_DATASET_LOCATION=<Path/to/WildCross>
_BOQ_CHECKPOINT_LOCATION=<Path/to/downloaded/pre-trained-urban-checkpoint-boq>
_EVAL_OUTPUT_LOCATION=<Path/to/save/eval/outputs>
_SPLIT_IDX=0

docker run --rm \
    -w /workspace/VPR \
    -v ${_WILDCROSS_DATASET_LOCATION}:/mnt/WildCross:ro \
    -v ${_BOQ_CHECKPOINT_LOCATION}:/mnt/urban_ckpts/BoQ_pretrained.pth \
    -v ${_EVAL_OUTPUT_LOCATION}:/mnt/eval_outputs\
    -e SPLIT_IDX=$_SPLIT_IDX \
    --gpus all \
    wildcross:latest /bin/bash -c \
    'eval "$(mamba shell hook --shell bash)" && 
    mamba activate WildCrossVPR && 
    export PYTHONPATH=$PWD:$PYTHONPATH && 
    python evaluation/eval_intra_sequence.py \
    --config configs/BoQ.yaml \
    --save_dir /mnt/eval_outputs/intra/zero-shot/BoQ/ \
    --pretrained_ckpt /mnt/urban_ckpts/BoQ_pretrained.pth \
    split_idx $SPLIT_IDX'