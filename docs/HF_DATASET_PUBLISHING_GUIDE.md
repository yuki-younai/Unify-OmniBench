# HuggingFace 数据集发布经验总结：上传 / 规范 / Viewer 可视化

> 背景：`yukiyounai/Unify-OmniBench` 数据集在 HF 页面上的 Dataset Viewer 长期只显示一列
> `audio`（`{"bytes": null, "path": "hf://..."}`），完全丢失了 `question`/`choices`/
> `answer`/`image` 等核心字段，而且每次改 `main` 分支之后又会"变回去"。本文记录排查过程
> 中搞清楚的原理，以及后续新增数据集/子集时应该怎么做才能一次做对，避免重复踩坑。

## 1. 问题本质：Dataset Viewer 显示的不是你上传的文件

HF 数据集页面的表格预览（Dataset Viewer）**默认并不直接读取你上传的原始文件**
（比如 `data/xxx.json` + `media/`），而是读取一个后台自动生成、和 `main` 平行的隐藏分支：

```
refs/convert/parquet
```

路径形如 `<config>/<split>/0000.parquet`（比如 `default/train/0000.parquet`）。这个分支由
HF 的 `dataset-viewer` 服务自动维护，规则是：

- **如果你的数据本身已经是规范的 Parquet 格式** → 不转换，`refs/convert/parquet` 里只是
  指向原始 Parquet 文件的"链接"，内容跟你传的一模一样。
- **如果你的数据不是 Parquet**（json/jsonl/csv/图片文件夹/音频文件夹等）→ 会用
  🤗 `datasets` 库的启发式规则（`imagefolder`/`audiofolder`/`json` builder 等）**自动猜测
  结构**，猜出一个 Parquet 版本发布到这个分支。

我们的仓库结构是：

```
data/omnibench.json          # 真正的结构化数据（question/choices/answer/...）
media/audio/omnibench/*.mp3  # 纯音频文件夹
media/image/omnibench/*.png  # 纯图片文件夹
```

当同时存在"结构化 JSON"和"媒体文件夹"时，HF 的自动结构探测会出现歧义，实际观察到的
结果是它**优先把 `media/audio/omnibench/` 识别成了一个独立的 AudioFolder 数据集**，完全
忽略了 `data/omnibench.json` 里的其它字段——这就是 Viewer 里只剩一列 `audio` 的根因。

而且只要 `main` 分支有新的 commit（哪怕只是改一下 README），HF 就会**重新跑一次这个自动
转换**，把手动改好的 `refs/convert/parquet` 内容又覆盖回简化版。所以"直接改
`refs/convert/parquet`"只能临时止血，不是根治办法。

## 2. 根治方案：让 `main` 分支本身就是规范的 Parquet

结论一句话：**把最终想要展示的规范数据，直接构建成一个 Parquet 文件提交进 `main` 分支，
并在 `README.md` 的 YAML frontmatter 里用 `configs` 显式声明它。**

因为 Parquet 格式不会被"重新转换"，只会被"链接"，所以只要这个文件本身是对的，
`refs/convert/parquet` 就会一直是对的，不会再被 main 分支上后续的其它 commit 打回原样。

### 2.1 README.md 需要的 YAML frontmatter

```yaml
---
language:
- en
license: other
task_categories:
- question-answering
- visual-question-answering
tags:
- multimodal
- omni
- audio
- image
- benchmark
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files:
  - split: train
    path: "data/train-*.parquet"
---
```

- **一定要有 frontmatter**，否则 HF 会报 `YAML Metadata Warning: empty or missing yaml
  metadata in repo card`（纯粹是文档规范问题，跟 Viewer 数据无关，但同样值得顺手修掉）。
- `configs.data_files.path` 精确指向你自己上传的规范 Parquet，避免 HF 再去猜
  `media/` 文件夹或其它 json 文件的结构。
- 命名建议遵循 `datasets` 库的默认可识别模式：`data/<split>-000xx-of-000yy.parquet`
  （如 `data/train-00000-of-00001.parquet`），即使不写 YAML，很多情况下也能被自动识别；
  但仓库里有多个候选文件/文件夹时，显式写 `configs` 更稳妥、不依赖猜测。

### 2.2 Parquet 里 Audio / Image 字段怎么写

**不需要把音频/图片真实二进制内容塞进 Parquet**（我们的媒体总大小 ~1.28GB，没必要也没
必要拖慢加载），只需要用 HF 的"外部引用"格式：

```json
{"bytes": null, "path": "hf://datasets/<repo_id>@<revision>/<相对路径>"}
```

- `bytes: null` 表示不内嵌数据；
- `path` 是一个 `hf://` 协议的 URI，`datasets` 库和 Dataset Viewer 都支持直接从这个
  URI 远程拉取文件用于播放/预览，不需要事先下载到本地。
- `<revision>` 建议用 **`main`**（moving ref）而不是写死某个 commit sha，只要媒体文件的
  相对路径不变，这个引用就一直有效，不用每次改动 main 分支后都重新生成 Parquet。

对应的 Arrow/Parquet schema 元数据，需要显式告诉 HF 这些列是 `Audio`/`Image` 类型
（不然只会显示成普通字符串路径，没有播放器/预览图）：

```python
import pyarrow as pa
import pyarrow.parquet as pq
import json

media_struct = pa.struct([("bytes", pa.binary()), ("path", pa.string())])

# ... 构造好各列的 pa.array 之后 ...
table = pa.Table.from_arrays(arrays, names=names)

hf_meta = {
    "info": {
        "features": {
            "audio_path": {"_type": "Audio"},
            "image_path": {"_type": "Image"},
        }
    }
}
table = table.replace_schema_metadata({"huggingface": json.dumps(hf_meta)})
pq.write_table(table, "train-00000-of-00001.parquet")
```

> **踩坑记录**：如果改用 `datasets` 库的 `Audio()`/`Image()` Feature 对象走
> `Dataset.from_list(...).to_parquet(...)`，`Audio.encode_example()` 在新版 `datasets`
> 里即使只是传一个字符串路径也会**无条件 `import torch` + `from torchcodec.encoders
> import AudioEncoder`**，环境里往往没装 `torchcodec`（体积大、还依赖 ffmpeg），会直接
> 报 `ImportError`。**绕过方法**：不用 `datasets` 库的 Feature 封装，直接用上面这种纯
> `pyarrow` 手工构造 struct 列 + schema metadata 的方式，效果完全一致，且零额外依赖。

### 2.3 一次性推送脚本骨架

```python
import json, os
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ID = "yukiyounai/Unify-OmniBench"
REF = "main"                      # 用 main 而不是写死 sha
JSON_PATH = ".../data/omnibench.json"
OUT_PATH = "/tmp/hf_work/train-00000-of-00001.parquet"

def hf_uri(rel_path):
    return f"hf://datasets/{REPO_ID}@{REF}/{rel_path}" if rel_path else None

records = json.load(open(JSON_PATH, encoding="utf-8"))
# ... 按 records 的字段构造 cols 字典 ...
# audio_path / image_path 两列用 {"bytes": None, "path": hf_uri(...)}

media_struct = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
table = pa.Table.from_arrays(arrays, names=names)
table = table.replace_schema_metadata({
    "huggingface": json.dumps({"info": {"features": {
        "audio_path": {"_type": "Audio"},
        "image_path": {"_type": "Image"},
    }}})
})
pq.write_table(table, OUT_PATH)
```

推送用 `huggingface_hub` 的 `HfApi`，可以把 Parquet 和更新后的 README 放在**同一个
commit**里一起提交：

```python
from huggingface_hub import HfApi, CommitOperationAdd

api = HfApi(token=HF_TOKEN)
ops = [
    CommitOperationAdd(path_in_repo="data/train-00000-of-00001.parquet",
                        path_or_fileobj=OUT_PATH),
    CommitOperationAdd(path_in_repo="README.md",
                        path_or_fileobj="README_v2.md"),
]
api.create_commit(
    repo_id=REPO_ID, repo_type="dataset", revision="main",
    operations=ops,
    commit_message="feat: add canonical parquet + declare configs",
)
```

如果只是想临时/单独覆盖 `refs/convert/parquet` 分支上的某个文件（治标不治本，适合应急）：

```python
api.upload_file(
    path_or_fileobj=OUT_PATH,
    path_in_repo="default/train/0000.parquet",
    repo_id=REPO_ID, repo_type="dataset",
    revision="refs/convert/parquet",
)
```

## 3. Token / 环境相关的坑

- `HfApi(token=...).whoami()` 是验证 token 是否有效最快的方式，401 大概率是 **token 已
  过期/被撤销**，不是权限或 repo_id 写错（错误信息里的 `RepositoryNotFoundError` 很容易
  误导，其实是认证失败导致连"仓库是否存在"都判断不出来）。
- 沙盒环境里 `python3`（3.6.8）和 `pip3` 对应的 `python3.12` 是两套完全独立的环境，
  `pip install` 装的包 `python3` 那边 import 不到。要用 `python3.12` 对应的 pip，或者
  直接用 `python3.12 -m pip install ...` / 显式调用 `python3.12` 跑脚本。
- `huggingface_hub` / `datasets` / `pyarrow` 装好之后，**不需要装 `torchcodec`/`torch`**
  就能完成"构造引用型 Audio/Image 列 + 写 Parquet + 上传"这整套流程（见上面 2.2 的绕过
  写法）。只有需要真正**内嵌音频二进制字节**（而不是引用）时才会用到 `torchcodec`。

## 4. README Repo Card 规范检查清单

新增/修改任意 HF 数据集时，建议顺手过一遍：

- [ ] 顶部有 YAML frontmatter（`---...---`），否则报 `YAML Metadata Warning`。
- [ ] `license` 字段（哪怕填 `other` 也比空着好）。
- [ ] `task_categories` / `tags` / `language` 尽量填，影响 Hub 搜索和分类展示。
- [ ] 如果仓库里同时存在"结构化数据文件"和"媒体文件夹"，一定要用 `configs.data_files`
  显式指向真正要展示的数据文件，不要依赖自动探测。
- [ ] 数据格式说明（字段列表、类型、示例 JSON）与实际数据保持一致——踩过的坑：README
  里写的是 `choices: list[str]` + `meta: dict` 嵌套结构，但实际 `data/omnibench.json`
  里是拍平的 `choice_a`~`choice_d` + `meta_audio_type`/`meta_audio_content`/
  `meta_image_content`，两者长期不一致，写脚本时以**实际数据**为准，同时顺手把文档改对。

## 5. 排查思路小结（下次遇到类似问题时的检查顺序）

1. 先用 `huggingface_hub.hf_hub_download` 把 `refs/convert/parquet` 分支上的实际
   `.parquet` 文件下下来，用 `pyarrow.parquet.read_table` 看 schema —— 这是"眼见为实"，
   比只看网页 Viewer 截图能确定得多。
2. 对比 schema 和 `main` 分支的原始数据字段，判断是"字段丢失"还是"类型没标对"
   （比如字符串路径没被标成 `Audio`/`Image`，Viewer 就不会渲染播放器/预览图）。
3. 检查 `main` 分支是不是已经是 Parquet 格式：如果不是，几乎可以确定
   `refs/convert/parquet` 会被自动转换规则反复覆盖，治标方案（直接改
   `refs/convert/parquet`）只能撑到下一次 main 分支 commit。
4. 根治：把规范 Parquet 提交进 `main`，README 加 `configs` 声明，让 HF 从"需要转换"
   变成"只需要链接"。
