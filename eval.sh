#!/usr/bin/env bash
# Unify-OmniBench launcher.
#
# Backends:
#   transformer  — local Transformers (需要 GPU，model-path 必填，backend=qwen_omni)
#   vllm         — local vLLM 离线   (需要 GPU，model-path 必填)
#   openai       — 纯 vLLM serve / GPT-4o 等标准 OpenAI API
#                  (服务端: bash vllm_deploy.sh)
#   openai-omni  — vllm-omni --omni 多阶段 pipeline server
#                  (服务端: bash vllm_omni_deploy.sh)
#   echo         — smoke-test (无 GPU / 无 API)
#
# Datasets: daily_omni | omnibench | omnivideobench
# DATASETS 支持数组，多个 bench 依次评测：
#   DATASETS=(daily_omni omnibench omnivideobench) bash eval.sh
#
# Modes:
#   norm — 直接输出答案字母 (默认)
#   cot  — Chain-of-Thought 推理模式（仅切换 prompt 文案，不再自动改 max_tokens）
# max_tokens 统一由 MAX_NEW_TOKENS 变量决定（不再受 INFER_MODE 隐式影响）：
#   - 留空 = 默认值 10（硬编码在 config/__init__.py::get_generation_cfg，不再依赖 yaml 文件）
#   - 想跑 cot 模式建议显式设置更大的值，比如 MAX_NEW_TOKENS=1024
# 结果目录: results/<dataset>/<model_name>_<backend>_<mode>/

export CUDA_VISIBLE_DEVICES=4,5,6,7   # eval 用 0-3，vllm_deploy.sh 用 4-7
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_DISABLE_PROGRESS_BAR=1

BACKEND=openai                                  # openai | openai-omni | vllm | transformer | echo
DATASETS=(omnibench omnivideobench)                                # 支持多个：DATASETS=(daily_omni omnibench)
INFER_MODE=norm                                       # norm | cot
MODEL_PATH=/apdcephfs_hldy/share_304318596/weiyangguo/models/Qwen2.5-Omni-7B    
MODEL_NAME=Qwen2.5-Omni-7B                          # results/<DATASET>/<MODEL_NAME>_<BACKEND>_<MODE>/
WORKERS=24                                           # api backend 并发线程数
API_URL=http://localhost:8001/v1                     # API server 地址
API_KEY=                                              # 空=本地vLLM / 非空=公有云(自动读$OPENAI_API_KEY)
TEMPERATURE=0.0                                         # 空 = 默认 (0.0)
TOP_P=                                               # 空 = 默认（不显式传）
MAX_NEW_TOKENS=512                                      # 空 = 默认 (10)；唯一决定 max_tokens 的开关

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
