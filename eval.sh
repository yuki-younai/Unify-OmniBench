#!/usr/bin/env bash
# Unify-OmniBench launcher.
#
# Backends: transformer | vllm (本地GPU) | openai | openai-omni (API/服务端) | echo (smoke-test)
# Datasets: daily_omni  omnibench  omnivideobench  worldsense（DATASETS=(a b) 可批量跑）
# Modes: norm (直出答案) | cot (思维链，建议配合更大的 MAX_NEW_TOKENS)
# 结果目录: results/<dataset>/<model_name>_<backend>_<mode>/
#
# 多 Worker 并行（vllm / transformer 后端）：
#   GPUS_PER_WORKER=0（默认）→ 所有 GPU 给一个 worker，等同于单进程
#   GPUS_PER_WORKER=2          → 4 张 GPU → 2 个 worker，每个独占 2 张 GPU
#   样本按 hash(uid) % num_workers 分配，跑完自动合并 shard 结果

#export CUDA_VISIBLE_DEVICES=4,5,6,7   
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_DISABLE_PROGRESS_BAR=1
# 让 vLLM spawn 出来的 worker 子进程也吃到 sitecustomize.py 的兼容性补丁
export PYTHONPATH="$(cd "$(dirname "$0")" && pwd):${PYTHONPATH:-}"

BACKEND=vllm                                 # openai | openai-omni | vllm | transformer | echo
DATASETS=(omnivideobench)                    # 支持多个：DATASETS=(daily_omni omnibench)
INFER_MODE=norm                              # norm | cot
MODEL_PATH=/apdcephfs_hldy/share_304318596/weiyangguo/models/Qwen2.5-Omni-7B    
MODEL_NAME=Qwen2.5-Omni-7B                   # results/<DATASET>/<MODEL_NAME>_<BACKEND>_<MODE>/
WORKERS=1                                    # batch_size；vllm 后端同时也是 max_num_seqs
API_URL=http://localhost:8001/v1             # API server 地址（openai 模式用）
API_KEY=                                     # 空=本地vLLM / 非空=公有云
TEMPERATURE=0.0                              # 空 = 默认 (0.0)
TOP_P=                                       # 空 = 默认
MAX_NEW_TOKENS=512                           # 空 = 默认 (10)

# ── 多 Worker 并行 ──
GPUS_PER_WORKER=2                            # 0 = 所有 GPU 给一个 worker
                                             # >0 = 每个 worker 独占 N 张 GPU

set -e
cd "$(dirname "$0")"
mkdir -p logs

# 解析 GPU ID 列表
GPU_LIST=(${CUDA_VISIBLE_DEVICES//,/ })
NUM_GPUS=${#GPU_LIST[@]}

for DATASET in "${DATASETS[@]}"; do
  echo "=== [$(date '+%H:%M:%S')] dataset=$DATASET mode=$INFER_MODE ==="

  # worker 数：GPU 后端按 GPUS_PER_WORKER 计算，其他后端固定 1
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
    python run.py \
      --backend "$BACKEND" \
      --dataset "$DATASET" \
      --model-name "$MODEL_NAME" \
      --mode "$INFER_MODE" \
      --workers "$WORKERS" \
      $SHARD_ARGS \
      --api-url "$API_URL" \
      ${MODEL_PATH:+--model-path "$MODEL_PATH"} \
      ${API_KEY:+--api-key "$API_KEY"} \
      ${TEMPERATURE:+--temperature "$TEMPERATURE"} \
      ${TOP_P:+--top-p "$TOP_P"} \
      ${MAX_NEW_TOKENS:+--max-new-tokens "$MAX_NEW_TOKENS"} \
      > "logs/${DATASET}_worker${i}.log" 2>&1 &
    PIDS+=($!)
  done

  for pid in "${PIDS[@]}"; do
    wait "$pid" || echo "WARNING: worker pid=$pid exited with code $?"
  done
  echo "=== [$(date '+%H:%M:%S')] all $NUM_WORKERS workers done for $DATASET ==="

  # 合并 shard 结果（仅多 worker 时需要）
  if $IS_GPU_BACKEND; then
    RESULT_DIR="results/$DATASET/${MODEL_NAME}_${BACKEND}_${INFER_MODE}"
    python3 script/merge_shards.py \
      --result-dir "$RESULT_DIR" \
      --num-shards "$NUM_WORKERS" \
      --dataset "$DATASET" \
      2>&1 | tee -a "logs/merge_${DATASET}.log"
  fi
done

echo "=== 全部完成: ${DATASETS[*]} ==="

# 自动生成 results/summary.md 聚合总表
python3 "$(dirname "$0")/script/aggregate_results.py"
