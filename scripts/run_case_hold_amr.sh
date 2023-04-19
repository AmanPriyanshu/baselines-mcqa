#!/bin/bash
#SBATCH --mem=30G
#SBATCH --time=5-0
#SBATCH -p gpu --gres=gpu:titanx:1
#SBATCH -c6

GPU_NUMBER=0
MODEL_NAME='/Users/niko/ML/case_hold/amrbart/model'
BATCH_SIZE=8
ACCUMULATION_STEPS=2
TASK='case_hold'
HOME_PATH='/Users/niko/ML/case_hold'
DATA_SET_PATH='/Users/niko/ML/case_hold/data/LogiQADataset/dataset_amr'

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} HOME_PATH=${HOME_PATH} python experiments/case_hold.py \
    --task_name ${TASK} --model_name_or_path ${MODEL_NAME} \
    --output_dir logs/${TASK}/${MODEL_NAME}/seed_1 \
    --do_train --do_eval --do_pred \
    --overwrite_cache \
    --overwrite_output_dir \
    --load_best_model_at_end \
    --metric_for_best_model micro-f1 \
    --greater_is_better True \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 5 \
    --num_train_epochs 20 \
    --learning_rate 3e-5 \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --seed 1 \
    --gradient_accumulation_steps ${ACCUMULATION_STEPS} \
    --eval_accumulation_steps ${ACCUMULATION_STEPS} \
    --is_amr \
    --data_set_path ${DATA_SET_PATH} \
    --max_seq_length 1024 \
