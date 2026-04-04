#!/usr/bin/env bash
# --------------------------------------------------------
# MEDiC Detection - Distributed Test Launcher
# --------------------------------------------------------

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29501}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DETECTION_DIR="$( dirname "$SCRIPT_DIR" )"

# Change to detection directory so imports work correctly
cd "$DETECTION_DIR"

PYTHONPATH="$DETECTION_DIR:$PYTHONPATH" python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    "$SCRIPT_DIR/test.py" \
    $CONFIG \
    $CHECKPOINT \
    --launcher pytorch \
    ${@:4}
