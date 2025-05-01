export CUDA_VISIBLE_DEVICES=0,1
# export NCCL_BUFFSIZE=1048576
model_path=$1
instruct_ds=$2
OUTPUT_DIR=$3
WORLD_SIZE=1
TQ_GPU_NUM=$4
train_batch_size=$5
eval_batch_size=$6
output_model=${OUTPUT_DIR}

echo ${WORLD_SIZE}

torchrun --nnodes=${WORLD_SIZE} --nproc_per_node=${TQ_GPU_NUM} --master_port=15668 ./tsgpt/train/train_mem.py \
    --model_name_or_path ${model_path} \
    --llm_version mpt \
    --st_version v0 \
    --data_path ${instruct_ds} \
    --lora_enable True \
    --tune_mlp_adapter True \
    --freeze_backbone True \
    --use_st_start_end \
    --output_dir ${output_model} \
    --num_train_epochs 3 \
    --per_device_train_batch_size ${train_batch_size} \
    --per_device_eval_batch_size ${eval_batch_size} \
    --gradient_accumulation_steps 4 \
    --num_nodes=${WORLD_SIZE} \
    --save_strategy "no" \
    --save_total_limit 1 \
    --learning_rate 3e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 4096 \
    --gradient_checkpointing False \
    --lazy_preprocess True \
    --strategy deepspeed_stage_2_offload \
    --precision bf16
