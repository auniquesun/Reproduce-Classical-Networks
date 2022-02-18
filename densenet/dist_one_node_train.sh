# distributed training on single node

GPUS=$1
local_rank=$2

python -m torch.distributed.launch --nproc_per_node=$GPUS \
            main_single.py 