CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

if [ -z "$GPUS" ]; then
    echo "Error: Please specify the number of GPUs."
    echo "Usage: bash dist_train.sh <CONFIG> <GPUS> [ARGS]"
    exit 1
fi

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/dist_train.py $CONFIG \
    --launcher pytorch \
    ${@:3}
