CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29500}

if [ -z "$GPUS" ]; then
    echo "Error: Please specify the number of GPUs."
    echo "Usage: bash dist_test.sh <CONFIG> <CHECKPOINT> <GPUS> [ARGS]"
    exit 1
fi

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/dist_test.py $CONFIG $CHECKPOINT \
    --launcher pytorch \
    ${@:4}
