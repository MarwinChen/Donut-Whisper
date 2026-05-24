#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")"; pwd)
cd "$PROJECT_ROOT"

WHISPER_PATH=${WHISPER_PATH:-openai/whisper-base}
WHISPER_LARGE_PATH=${WHISPER_LARGE_PATH:-openai/whisper-large-v3}
IMAGE_MODEL_PATH=${IMAGE_MODEL_PATH:-naver-clova-ix/donut-base-finetuned-cord-v2}
TRAIN_DATA=${TRAIN_DATA:-}
EVAL_DATA=${EVAL_DATA:-}
TEST_DATA=${TEST_DATA:-}

EPOCHS=${EPOCHS:-50}
TRAIN_BS=${TRAIN_BS:-16}
EVAL_BS=${EVAL_BS:-2}
LR=${LR:-1e-4}
SEED=${SEED:-2025}
SAVE_STEPS=${SAVE_STEPS:-1000}
OUTPUT_NAME=${OUTPUT_NAME:-debug}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-8}
EVAL_ACCUMULATION_STEPS=${EVAL_ACCUMULATION_STEPS:--1}

DO_TEST=False
CKPT=${CKPT:-None}
CKPT_FOR_DIS=${CKPT_FOR_DIS:-None}
MODEL_TYPE=${MODEL_TYPE:-donut_whisper}
TRAIN_ENCODER=${TRAIN_ENCODER:-False}
USE_LORA=${USE_LORA:-True}
FP16=${FP16:-True}
DEEPSPEED=${DEEPSPEED:-scripts/zero0.json}

TOKENIZER_LANGUAGE=${TOKENIZER_LANGUAGE:-zh}
TOKENIZER_TASK=${TOKENIZER_TASK:-transcribe}
FUSION_TYPE=${FUSION_TYPE:-sliding_window_q_former}
FUSION_WINDOW_SIZE=${FUSION_WINDOW_SIZE:-64}
FUSION_STRIDE=${FUSION_STRIDE:-32}
FUSION_OVERLAP=${FUSION_OVERLAP:-True}
FUSION_NUM_HEADS=${FUSION_NUM_HEADS:-8}
FUSION_DROPOUT=${FUSION_DROPOUT:-0.1}

NPROC_PER_NODE=${NPROC_PER_NODE:-${ARNOLD_WORKER_GPU:-1}}
NNODES=${NNODES:-${ARNOLD_WORKER_NUM:-1}}
NODE_RANK=${NODE_RANK:-${ARNOLD_ID:-0}}
MASTER_ADDR=${MASTER_ADDR:-${METIS_WORKER_0_HOST:-localhost}}
MASTER_PORT=${MASTER_PORT:-12397}
CUDA_VISIBLE_DEVICES_ARG=${CUDA_VISIBLE_DEVICES_ARG:-}

WANDB_PROJECT=${WANDB_PROJECT:-asr_ocr_whisper_dis}
WANDB_MODE=${WANDB_MODE:-offline}
HF_HOME_ARG=${HF_HOME_ARG:-}

usage() {
    cat <<'USAGE'
Usage:
  bash run.sh --train_data TRAIN.json --eval_data VAL.json [options]
  bash run.sh --do_test --test_data TEST.json --ckpt CKPT.bin [options]

Common options:
  --whisper_path PATH_OR_ID
  --whisper_large_path PATH_OR_ID
  --image_model_path PATH_OR_ID
  --model_type donut_whisper|donut_whisper_attn|whisper_only|donut_only|distillation_whisper_only|donut_whisper_ed
  --output_name NAME
  --epochs N
  --train_bs N
  --eval_bs N
  --lr FLOAT
  --save_steps N
  --nproc_per_node N
  --cuda_visible_devices IDS
  --deepspeed PATH_OR_None
  --fp16 True|False
USAGE
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --whisper_path) WHISPER_PATH="$2"; shift ;;
        --whisper_large_path) WHISPER_LARGE_PATH="$2"; shift ;;
        --image_model_path) IMAGE_MODEL_PATH="$2"; shift ;;
        --train_data) TRAIN_DATA="$2"; shift ;;
        --eval_data) EVAL_DATA="$2"; shift ;;
        --epochs) EPOCHS="$2"; shift ;;
        --train_bs) TRAIN_BS="$2"; shift ;;
        --eval_bs) EVAL_BS="$2"; shift ;;
        --lr) LR="$2"; shift ;;
        --seed) SEED="$2"; shift ;;
        --save_steps) SAVE_STEPS="$2"; shift ;;
        --output_name) OUTPUT_NAME="$2"; shift ;;
        --test_data) TEST_DATA="$2"; shift ;;
        --do_test) DO_TEST=True ;;
        --ckpt) CKPT="$2"; shift ;;
        --ckpt_for_dis) CKPT_FOR_DIS="$2"; shift ;;
        --model_type) MODEL_TYPE="$2"; shift ;;
        --train_encoder) TRAIN_ENCODER="$2"; shift ;;
        --use_lora) USE_LORA="$2"; shift ;;
        --fp16) FP16="$2"; shift ;;
        --deepspeed) DEEPSPEED="$2"; shift ;;
        --eval_accumulation_steps) EVAL_ACCUMULATION_STEPS="$2"; shift ;;
        --dataloader_num_workers) DATALOADER_NUM_WORKERS="$2"; shift ;;
        --nproc_per_node) NPROC_PER_NODE="$2"; shift ;;
        --nnodes) NNODES="$2"; shift ;;
        --node_rank) NODE_RANK="$2"; shift ;;
        --master_addr) MASTER_ADDR="$2"; shift ;;
        --master_port) MASTER_PORT="$2"; shift ;;
        --cuda_visible_devices) CUDA_VISIBLE_DEVICES_ARG="$2"; shift ;;
        --hf_home) HF_HOME_ARG="$2"; shift ;;
        --wandb_project) WANDB_PROJECT="$2"; shift ;;
        --wandb_mode) WANDB_MODE="$2"; shift ;;
        --tokenizer_language) TOKENIZER_LANGUAGE="$2"; shift ;;
        --tokenizer_task) TOKENIZER_TASK="$2"; shift ;;
        --fusion_type) FUSION_TYPE="$2"; shift ;;
        --fusion_window_size) FUSION_WINDOW_SIZE="$2"; shift ;;
        --fusion_stride) FUSION_STRIDE="$2"; shift ;;
        --fusion_overlap) FUSION_OVERLAP="$2"; shift ;;
        --fusion_num_heads) FUSION_NUM_HEADS="$2"; shift ;;
        --fusion_dropout) FUSION_DROPOUT="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage; exit 1 ;;
    esac
    shift
done

if [[ "$DO_TEST" == "True" ]]; then
    if [[ -z "$TEST_DATA" ]]; then
        echo "Error: --test_data is required with --do_test."
        exit 1
    fi
    DEEPSPEED=None
    OUTPUT_DIR="output/test/$OUTPUT_NAME"
else
    if [[ -z "$TRAIN_DATA" || -z "$EVAL_DATA" ]]; then
        echo "Error: --train_data and --eval_data are required for training."
        exit 1
    fi
    OUTPUT_DIR="output/$OUTPUT_NAME"
fi

if [[ -n "$CUDA_VISIBLE_DEVICES_ARG" ]]; then
    export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_ARG"
fi

if [[ -n "$HF_HOME_ARG" ]]; then
    export HF_HOME="$HF_HOME_ARG"
fi

export WANDB_PROJECT
export WANDB_NAME="$OUTPUT_DIR"
export WANDB_MODE

TRAIN_ARGS=(
    train.py
    --model_type "$MODEL_TYPE"
    --whisper_path "$WHISPER_PATH"
    --whisper_large_path "$WHISPER_LARGE_PATH"
    --image_model_path "$IMAGE_MODEL_PATH"
    --seed "$SEED"
    --num_train_epochs "$EPOCHS"
    --per_device_train_batch_size "$TRAIN_BS"
    --per_device_eval_batch_size "$EVAL_BS"
    --evaluation_strategy "steps"
    --save_total_limit 100
    --learning_rate "$LR"
    --weight_decay 0.
    --warmup_ratio 0.03
    --lr_scheduler_type "cosine"
    --logging_steps 1
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS"
    --eval_steps "$SAVE_STEPS"
    --save_steps "$SAVE_STEPS"
    --output_dir "$OUTPUT_DIR"
    --remove_unused_columns False
    --fp16 "$FP16"
    --test_data "$TEST_DATA"
    --do_test "$DO_TEST"
    --ckpt "$CKPT"
    --ckpt_for_dis "$CKPT_FOR_DIS"
    --train_encoder "$TRAIN_ENCODER"
    --use_lora "$USE_LORA"
    --tokenizer_language "$TOKENIZER_LANGUAGE"
    --tokenizer_task "$TOKENIZER_TASK"
    --fusion_type "$FUSION_TYPE"
    --fusion_window_size "$FUSION_WINDOW_SIZE"
    --fusion_stride "$FUSION_STRIDE"
    --fusion_overlap "$FUSION_OVERLAP"
    --fusion_num_heads "$FUSION_NUM_HEADS"
    --fusion_dropout "$FUSION_DROPOUT"
)

if [[ "$DO_TEST" != "True" ]]; then
    TRAIN_ARGS+=(--train_data "$TRAIN_DATA" --eval_data "$EVAL_DATA")
fi

if [[ "$DEEPSPEED" != "None" && "$DEEPSPEED" != "none" && -n "$DEEPSPEED" ]]; then
    TRAIN_ARGS+=(--deepspeed "$DEEPSPEED")
fi

if [[ "$EVAL_ACCUMULATION_STEPS" =~ ^-?[0-9]+$ && "$EVAL_ACCUMULATION_STEPS" -ge 0 ]]; then
    TRAIN_ARGS+=(--eval_accumulation_steps "$EVAL_ACCUMULATION_STEPS")
fi

echo "Output dir: $OUTPUT_DIR"
echo "Model type: $MODEL_TYPE"
echo "Whisper: $WHISPER_PATH"
echo "Image model: $IMAGE_MODEL_PATH"

torchrun \
    --nproc_per_node="$NPROC_PER_NODE" \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    "${TRAIN_ARGS[@]}"
