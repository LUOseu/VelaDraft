#!/bin/bash
exec 9>/tmp/vispec_sqa.lock
flock -n 9 || { echo "another instance is running"; exit 1; }
cd /home/lyh/ViSpec-reference
export PYTHONPATH=/home/lyh/ViSpec-reference
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1
PY=/home/lyh/ViSpec-reference/venv/bin/python
BASE=/home/lyh/vlm-dflash/assets/models/Qwen2.5-VL-3B-Instruct
SPEC=/home/lyh/ViSpec-reference/ckpt_qwen3b
DATA=/home/lyh/ViSpec-reference/scienceqa_data
OUT=/home/lyh/ViSpec-reference/eval_runs
LOG=/home/lyh/ViSpec-reference/eval_runs/vispec_sqa.log
mkdir -p $OUT
echo "=== baseline start $(date +%T) ===" >> $LOG
CUDA_VISIBLE_DEVICES=0 $PY -m vispec.evaluation.gen_baseline_answer_sqa \
  --base-model-path $BASE --spec-model-path $SPEC --model-id qwen3b_t0 \
  --bench-name $OUT/sqa_baseline --data_root $DATA --caption_file $DATA/captions.json \
  --test_split test --test_number -1 --shot_number 0 --prompt_format QCM-ALE \
  --temperature 0.0 >> $LOG 2>&1
echo "=== baseline rc=$? $(date +%T) ===" >> $LOG
echo "=== spec start $(date +%T) ===" >> $LOG
CUDA_VISIBLE_DEVICES=0 $PY -m vispec.evaluation.gen_spec_answer_sqa \
  --base-model-path $BASE --spec-model-path $SPEC --model-id qwen3b_t0 \
  --bench-name $OUT/sqa_spec --data_root $DATA --caption_file $DATA/captions.json \
  --test_split test --test_number -1 --shot_number 0 --prompt_format QCM-ALE \
  --temperature 0.0 \
  --use-ours True --num-q 2 --depth 3 --top-k 8 --total-token 30 >> $LOG 2>&1
echo "=== spec rc=$? $(date +%T) ===" >> $LOG
echo VISSPEC SQA DONE $(date +%T) >> $LOG
