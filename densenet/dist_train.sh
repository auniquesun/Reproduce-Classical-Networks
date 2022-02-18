GPUS=$1
PORT=${PORT:-29500}

echo $(@:3)
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/main.py  --launcher pytorch ${@:3}