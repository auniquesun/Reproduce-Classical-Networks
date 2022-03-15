# distributed training on single node

GPUS=$1
FREQ=$2
BATCH=$3
DATASET_ROOT=$4

python -m torch.distributed.launch --nproc_per_node=$GPUS \
            main_single.py --tensorboard --print_freq $FREQ --batch_size $BATCH --dataset_root $DATASET_ROOT
