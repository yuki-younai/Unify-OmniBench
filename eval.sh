#!/usr/bin/env bash
# Unify-OmniBench launcher.
#
# Backends: transformer | vllm (本地GPU) | openai | openai-omni (API/服务端) | echo (smoke-test)
# Datasets: daily_omni  omnibench  omnivideobench  worldsense videomme（DATASETS=(a b) 可批量跑）
# Modes: norm (直出答案) | cot (思维链，建议配合更大的 MAX_NEW_TOKENS)
# 结果目录: results/<dataset>/<model_name>_<backend>_<mode>/
#
# 多 Worker 并行（vllm / transformer 后端）：
#   GPUS_PER_WORKER=0（默认）→ 所有 GPU 给一个 worker，等同于单进程
#   GPUS_PER_WORKER=2          → 4 张 GPU → 2 个 worker，每个独占 2 张 GPU
#   样本按 hash(uid) % num_workers 分配，跑完自动合并 shard 结果

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7   

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_DISABLE_PROGRESS_BAR=1
export PYTHONPATH="$(cd "$(dirname "$0")" && pwd):${PYTHONPATH:-}"
PYTHON=${PYTHON:-python3.12}

BACKEND=transformer                               # openai | openai-omni | vllm | transformer | echo
DATASETS=(daily_omni  omnibench omnivideobench  worldsense videomme)                    # 支持多个：DATASETS=(daily_omni omnibench)
INFER_MODE=norm                              # norm | cot
RUN_MODE=direct                              # direct | react
MODEL_PATH=/apdcephfs_hldy/share_304318596/weiyangguo/models/Qwen2.5-Omni-7B    
MODEL_NAME=Qwen2.5-Omni-7B                   # results/<DATASET>/<MODEL_NAME>_<BACKEND>_<MODE>/
WORKERS=1                                    # batch_size；vllm 后端同时也是 max_num_seqs
API_URL=http://localhost:8001/v1             # API server 地址（openai 模式用）
API_KEY=                                     # 空=本地vLLM / 非空=公有云
TEMPERATURE=0.0                              # 空 = 默认 (0.0)
TOP_P=                                       # 空 = 默认
MAX_NEW_TOKENS=512                           # 空 = 默认 (10)

# ── Multi-Worker ──
GPUS_PER_WORKER=1                            # 0 = all GPUs for one worker
                                             # >0 = N GPUs per worker

set -eo pipefail
cd "$(dirname "$0")"

# parse GPU ID list
GPU_LIST=(${CUDA_VISIBLE_DEVICES//,/ })
NUM_GPUS=${#GPU_LIST[@]}

for DATASET in "${DATASETS[@]}"; do
  echo "=== [$(date '+%H:%M:%S')] dataset=$DATASET mode=$INFER_MODE ==="

  RESULT_DIR="results/$DATASET/${MODEL_NAME}_${BACKEND}_${INFER_MODE}"

  # skip if already completed
  if SKIP_MSG=$($PYTHON script/check_completed.py --result-dir "$RESULT_DIR" 2>/dev/null); then
    echo "  [skip] already completed (accuracy=$SKIP_MSG), no failed samples — skipping"
    continue
  fi

  mkdir -p "$RESULT_DIR"

  # worker count: GPU backends use GPUS_PER_WORKER, others fixed to 1
  IS_GPU_BACKEND=false
  if { [ "$BACKEND" = "vllm" ] || [ "$BACKEND" = "transformer" ]; }; then
    IS_GPU_BACKEND=true
    [ "$GPUS_PER_WORKER" -gt 0 ] || GPUS_PER_WORKER=$NUM_GPUS
    NUM_WORKERS=$(( NUM_GPUS / GPUS_PER_WORKER ))
    [ "$NUM_WORKERS" -ge 1 ] || { echo "ERROR: NUM_GPUS=$NUM_GPUS < GPUS_PER_WORKER=$GPUS_PER_WORKER"; exit 1; }
    echo "Multi-worker: $NUM_WORKERS workers × $GPUS_PER_WORKER GPU(s) each"
  else
    NUM_WORKERS=1
  fi

  PIDS=()
  for ((i=0; i<NUM_WORKERS; i++)); do
    SHARD_ENV=""
    SHARD_ARGS=""
    if $IS_GPU_BACKEND; then
      START=$(( i * GPUS_PER_WORKER ))
      SHARD_GPUS="${GPU_LIST[*]:$START:$GPUS_PER_WORKER}"
      SHARD_GPUS="${SHARD_GPUS// /,}"
      SHARD_ENV="CUDA_VISIBLE_DEVICES=$SHARD_GPUS"
      SHARD_ARGS="--shard-id $i --num-shards $NUM_WORKERS"
      echo "  [worker $i/$NUM_WORKERS] $SHARD_ENV"
    fi

    env $SHARD_ENV \
    $PYTHON run.py \
      --backend "$BACKEND" \
      --dataset "$DATASET" \
      --model-name "$MODEL_NAME" \
      --mode "$INFER_MODE" \
      --run-mode "$RUN_MODE" \
      --workers "$WORKERS" \
      $SHARD_ARGS \
      --api-url "$API_URL" \
      ${MODEL_PATH:+--model-path "$MODEL_PATH"} \
      ${API_KEY:+--api-key "$API_KEY"} \
      ${TEMPERATURE:+--temperature "$TEMPERATURE"} \
      ${TOP_P:+--top-p "$TOP_P"} \
      ${MAX_NEW_TOKENS:+--max-new-tokens "$MAX_NEW_TOKENS"} \
      > "$RESULT_DIR/worker${i}.log" 2>&1 &
    PIDS+=($!)
  done

  for pid in "${PIDS[@]}"; do
    # `set -e` would kill the script if wait returns non-zero.
    # Use `|| true` to suppress that, then check the real exit code.
    wait "$pid" && continue
    echo "WARNING: worker pid=$pid exited with code $?"
  done
  echo "=== [$(date '+%H:%M:%S')] all $NUM_WORKERS workers done for $DATASET ==="

  # merge shard results (multi-worker only)
  if $IS_GPU_BACKEND; then
    $PYTHON script/merge_shards.py \
      --result-dir "$RESULT_DIR" \
      --num-shards "$NUM_WORKERS" \
      --dataset "$DATASET" \
      --cleanup \
      2>&1 | tee -a "$RESULT_DIR/merge.log"
  fi
done

echo "=== All done: ${DATASETS[*]} ==="

# auto-generate results/summary.md
$PYTHON "$(dirname "$0")/script/aggregate_results.py"
