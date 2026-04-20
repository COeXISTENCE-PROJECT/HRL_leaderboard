#!/bin/bash

cd /app/

python3 -m venv /tmp/venv
source /tmp/venv/bin/activate

echo "--- Installing dependencies ---"
pip3 install --force-reinstall --no-cache-dir -r requirements.txt



echo "--- Running URB experiment ---"

# -------------------------------
# Define URB experiment variables
# -------------------------------
ALGORITHM="qmix_torchrl"
CITY="saint_arnoult"	# City


# Configs (algorithm, environment, task)
ALG_CONF="test"			
ENV_CONF="test"
TASK_CONF="test"

# Seeds
ENV_SEED=42
TORCH_SEED=0

# Create experiment ID
DATE_STR=$(date +%Y%m%d_%H%M%S)
EXP_ID="${ALGORITHM}_${CITY}_${DATE_STR}"	# Define experiment name


# -------------------------------
# Run URB experiment
# -------------------------------
# python3 -u scripts/ippo_torchrl.py --id 1 --conf 1_ippo --net gargenville --seed 42
python3 -u scripts/${ALGORITHM}.py \
    --id ${EXP_ID} \
    --alg-conf ${ALG_CONF} \
    --env-conf ${ENV_CONF} \
    --task-conf ${TASK_CONF} \
    --net ${CITY} \
    --env-seed ${ENV_SEED} \
    --torch-seed ${TORCH_SEED}

