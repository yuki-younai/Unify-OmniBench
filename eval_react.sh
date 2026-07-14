#!/usr/bin/env bash
# Unify-OmniBench launcher (Agent ReAct mode).
#
# Backends: transformer | vllm (local GPU) | openai | openai-omni (API) | echo (smoke-test)
# Datasets: daily_omni  omnibench  omnivideobench  worldsense  videomme
# ReAct config: config/agent.yaml (max_steps, tools, etc.)
# Output dir: results/<dataset>/<model_name>_<backend>_<mode>/

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_DISABLE_PROGRESS_BAR=1
export PYTHONPATH="$(cd "$(dirname "$0")" && pwd):${PYTHONPATH:-}"
PYTHON=${PYTHON:-python3.12}

# relax get_audio/get_clip ffprobe duration-tolerance check (see tools.py::VideoEnv)
# — matches OmniAgent's examples/omniagent_eval/eval.sh
export BYPASS_DURATION_CHECK=True

BACKEND=vllm
DATASETS=(daily_omni  omnibench omnivideobench  worldsense videomme future_omni)
INFER_MODE=norm
RUN_MODE=react

#/apdcephfs_hldy/share_304318596/weiyangguo/models/Qwen2.5-Omni-3B
#/apdcephfs_hldy/share_304318596/weiyangguo/models/OmniAgent-RL-7B
MODEL_PATH=/apdcephfs_hldy/share_304318596/weiyangguo/models/OmniAgent-RL-7B
MODEL_NAME=OmniAgent-RL-7B
WORKERS=1                                    # vllm max_num_seqs (react benefits from higher concurrency)
API_URL=http://localhost:8001/v1
API_KEY=
TEMPERATURE=0.0
TOP_P=
MAX_NEW_TOKENS=4096                           # react needs more tokens for multi-turn interaction

# ── Multi-Worker ──
GPUS_PER_WORKER=1                            # 0 = all GPUs for one worker, >0 = N GPUs per worker

set -eo pipefail
cd "$(dirname "$0")"

# cleanup leftover GPU processes
fuser -k /dev/nvidia* 2>/dev/null || true
sleep 2

# parse GPU ID list
GPU_LIST=(${CUDA_VISIBLE_DEVICES//,/ })
NUM_GPUS=${#GPU_LIST[@]}

for DATASET in "${DATASETS[@]}"; do
  echo "=== [$(date '+%H:%M:%S')] dataset=$DATASET mode=$INFER_MODE run=$RUN_MODE ==="

  RESULT_DIR="results/$DATASET/${MODEL_NAME}_${BACKEND}_${INFER_MODE}_react"

  # skip if already completed
  if SKIP_MSG=$($PYTHON script/check_completed.py --result-dir "$RESULT_DIR" 2>/dev/null); then
    echo "  [skip] already completed (accuracy=$SKIP_MSG) — skipping"
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
