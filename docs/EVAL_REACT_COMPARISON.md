# eval_react.sh vs OmniAgent eval.sh 对比

## 整体流程对比

| 步骤 | OmniAgent | Unify-OmniBench eval_react.sh | 一致? |
|---|---|---|---|
| 1. GPU 清理 | `fuser -k /dev/nvidia*` + pkill | ❌ 无 | ⚠️ 缺失 |
| 2. 环境变量 | `VLLM_WORKER_MULTIPROC_METHOD=spawn` 等 | 同 | ✅ |
| 3. 解析 GPU ID | 直接从 `CUDA_VISIBLE_DEVICES` 传到 `--gpu_ids` | `GPU_LIST=(${CUDA_VISIBLE_DEVICES//,/ })` 手动拆 | ✅ 等价 |
| 4. 多 Worker 启动 | `multiprocessing.spawn` 进程池, 主进程 `CUDA=""` | bash `&` 后台并发 + `wait` | ✅ 效果一致 |
| 5. 样本分配 | `task_queue` 竞争消费 | `hash(uid) % num_workers` 静态分片 | ✅ |
| 6. 结果收集 | `result_queue` → 主进程写 `results.jsonl` | 各 worker 独立写 `shard_N/items.jsonl` → merge | ✅ |
| 7. 断点续跑 | 从 `results.jsonl` 读取已完成 uid | 从 `items.jsonl` 读取已完成 uid | ✅ |
| 8. 评测循环 | `while not done: generate → parse → execute → append` | 同（ReActEvaluator） | ✅ |
| 9. 多数据集 | 单数据集 / 空格分隔列表 | 同 `DATASETS=(a b)` | ✅ |
| 10. 日志 | `eval_logs/YYYYMMDD_HHMMSS/` | `results/dataset/model_backend_mode/workerN.log` | ✅ 都有 |

## 参数配置对比

### Agent 核心参数

| 参数 | OmniAgent | eval_react.sh | 状态 |
|---|---|---|---|
| **max_steps** | 32 (可配 `MAX_STEPS`) | 32 (`dataset_config.yaml` 的 `react.max_steps`) | ✅ |
| **temperature** | **1.0** | **0.0** | ❌ **差异巨大** |
| **top_p** | 0.95 | 未设（默认 1.0） | ⚠️ 不同 |
| **top_k** | 20 | 未设 | ⚠️ 缺失 |
| **max_model_len** | 65536 | 32768 (vllm.yaml 默认) | ❌ Agent 需要更长上下文 |
| **max_prompt_len** | 32768 | 未显式设 | ⚠️ |
| **max_response_len** | 4096 | 2048 (react config) | ⚠️ 偏小 |

### 工具限制参数

| 参数 | OmniAgent | eval_react.sh | 状态 |
|---|---|---|---|
| **max_frames_len** (get_frames 每轮最多抽帧数) | 60 | ❌ 未设（默认 10） | ❌ **缺失** |
| **max_audio_len** (get_audio 最长秒数) | 300s | ❌ 未设 | ❌ **缺失** |
| **max_clip_len** (get_clip 最长秒数) | 60s | ❌ 未设 | ❌ **缺失** |

### 高级特性

| 特性 | OmniAgent | eval_react.sh | 状态 |
|---|---|---|---|
| **Dynamic Step** (步数按视频时长自适应) | ✅ `USE_DYNAMIC_STEP=true` | ❌ | ❌ 缺失 |
| **TITO** (Think-in-Turn-Out) | `USE_TITO=false` | N/A | 暂不需要 |
| **显存** | `gpu_memory_util=0.7` | 0.95 (vllm.yaml) | ❌ Agent 累积媒体需更多显存 |

## 关键差异总结

### 🔴 必须修复

1. **temperature=0.0 → 需改为 1.0**  
   Agent 模式需要 temperature > 0 才能产生多样化的工具选择（探索 vs 直接 answer）。temperature=0 会让模型每次都走同一条路径，失去 Agent 探索的意义。  
   
   建议：在 `dataset_config.yaml` 的 `react.generation` 中设 `temperature: 1.0`。

2. **max_model_len**  
   多轮对话 + 累积媒体 token 远超单次推理。vllm.yaml 默认 32768 不够。  
   建议：`eval_react.sh` 中新增 `--max-model-len` 覆盖，或 react 配置中 `max_model_len: 65536`。

### 🟡 建议补充

3. **max_frames_len / max_audio_len / max_clip_len**  
   控制单轮工具调用的媒体量上限。get_frames 默认 10 帧，OmniAgent 允许到 60。  
   建议：在 `get_frames` 的 execute 中读取 `react.max_frames_len` 配置上限。

4. **Dynamic Step**  
   视频越长需要越多探索步数。OmniAgent: `steps = min(5 + duration/max_clip_len, max_steps)`。  
   建议：在 `ReActEvaluator` 中按视频时长自适应 `max_steps`。

5. **GPU 清理**  
   OmniAgent 启动前 kill 残留 GPU 进程。  
   建议：`eval_react.sh` 增加 `fuser -k /dev/nvidia* 2>/dev/null || true`。

### 🟢 一致（无需改动）

- ✅ 多 worker 并行模式架构一致
- ✅ 断点续跑逻辑一致
- ✅ 样本分片（竞争消费 / hash 分片）效果等价
- ✅ Agent 循环 4 步完全一致（generate → parse → execute → append）
- ✅ 工具系统（get_frames / get_audio / get_clip / answer）命令一致
- ✅ eval.sh 参数传递流一致
