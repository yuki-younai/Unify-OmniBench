#!/usr/bin/env bash
# Unify-OmniBench launcher.
#
# Backends: transformer | vllm (本地GPU) | openai | openai-omni (API/服务端) | echo (smoke-test)
# Datasets: daily_omni  omnibench  omnivideobench  worldsense（DATASETS=(a b) 可批量跑）
# Modes: norm (直出答案) | cot (思维链，建议配合更大的 MAX_NEW_TOKENS)
# 结果目录: results/<dataset>/<model_name>_<backend>_<mode>/

export CUDA_VISIBLE_DEVICES=4,5,6,7   
#export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_DISABLE_PROGRESS_BAR=1
# 让 vLLM spawn 出来的 worker 子进程也吃到 sitecustomize.py 的兼容性补丁
export PYTHONPATH="$(cd "$(dirname "$0")" && pwd):${PYTHONPATH:-}"

BACKEND=vllm                                 # openai | openai-omni | vllm | transformer | echo
DATASETS=(worldsense)                                # 支持多个：DATASETS=(daily_omni omnibench)
INFER_MODE=norm                                       # norm | cot
MODEL_PATH=/apdcephfs_hldy/share_304318596/weiyangguo/models/Qwen2.5-Omni-3B    
MODEL_NAME=Qwen2.5-Omni-3B                          # results/<DATASET>/<MODEL_NAME>_<BACKEND>_<MODE>/
WORKERS=2                                             # batch_size；vllm 后端同时也是 max_num_seqs
                                                      # （引擎真实并发上限），显存紧张就调小
API_URL=http://localhost:8001/v1                     # API server 地址
API_KEY=                                              # 空=本地vLLM / 非空=公有云(自动读$OPENAI_API_KEY)
TEMPERATURE=0.0                                         # 空 = 默认 (0.0)
TOP_P=                                               # 空 = 默认（不显式传）
MAX_NEW_TOKENS=512                                      # 空 = 默认 (10)

set -e
cd "$(dirname "$0")"

for DATASET in "${DATASETS[@]}"; do
  echo "=== [$(date '+%H:%M:%S')] dataset=$DATASET mode=$INFER_MODE ==="
  python run.py \
    --backend "$BACKEND" \
    --dataset "$DATASET" \
    --model-name "$MODEL_NAME" \
    --mode "$INFER_MODE" \
    --workers "$WORKERS" \
    --api-url "$API_URL" \
    ${MODEL_PATH:+--model-path "$MODEL_PATH"} \
    ${API_KEY:+--api-key "$API_KEY"} \
    ${TEMPERATURE:+--temperature "$TEMPERATURE"} \
    ${TOP_P:+--top-p "$TOP_P"} \
    ${MAX_NEW_TOKENS:+--max-new-tokens "$MAX_NEW_TOKENS"}
done

echo "=== 全部完成: ${DATASETS[*]} ==="

# 自动生成 results/summary.md 聚合总表
python3 "$(dirname "$0")/script/aggregate_results.py"
