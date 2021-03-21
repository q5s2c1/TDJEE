#! /bin/bash

NUM_GPUS=$1
shift

python -m torch.distributed.launch --nproc_per_node ${NUM_GPUS} dee_predict_result.py $*
