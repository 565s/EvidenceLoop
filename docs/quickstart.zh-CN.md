# 中文入门

EvidenceLoop 接收完整文档，输出三元组及逐阶段审核记录。此次整理保留原实现的算法、提示词、解析器和回退行为；原项目、完整数据、权重及历史运行保留在原位置。本目录可供本地安装、审阅和后续开源，尚未发布到 GitHub。

## 先完成离线安装与查看

建议使用自己的 Python 3.10+ 虚拟环境，在新项目根目录运行：

```sh
python -m pip install -e .
python -m evidenceloop --help
evidenceloop data validate examples/documents.jsonl
```

上面三步不会调用模型。`examples` 只有人工编写的两条小示例，不能用于论文指标比较。输入采用 JSON 数组或 JSONL：

```json
{"document_id":"example-1","text":"Ada Lovelace was born in London."}
```

标识别名为 `article_id`、`sample_id`、`source_id`、`id`、`_id`；正文别名为 `article`、`source_text`、`context`。`title` 忽略，文档不自动切块。历史 `.source` 文件仍可读取，旧 `.source/.target` 配对只在数据处理和独立评测时使用。

## 准备真实推理

安装 `python -m pip install -e ".[critic]"`，准备支持 BF16 的 CUDA 环境、本地 Qwen3-8B 基座和对应数据集的 LoRA `adapter` 目录。四份 `configs/*.yaml` 提供完整字段模板，路径需按实际位置修改。未知权重下载地址未填写，也不会自动下载。

复制 `.env.example` 到私人文件填写 `OPENAI_BASE_URL`、`OPENAI_API_KEY`，或在 shell 设置环境变量。模板不带默认服务地址；历史 `SIFLOW_BASE_URL`、`SIFLOW_API_KEY` 仍可作为读取别名。不要把真实密钥纳入版本控制。

以下命令会实际请求 Proposer/Integrator，并加载 GPU Critic：

```sh
evidenceloop run --input examples/documents.jsonl --config configs/rebel_sub.yaml --env-file .env --output-run-dir runs/example
```

所有相对路径均相对于调用时的工作目录；在项目之外运行时请传绝对路径。初始和最终 Critic 共用同一个进程内安全合并的 BF16 模型。推理只读文档，不读 `gold_triples.jsonl`。

仅初始 Critic 变体示例：

```sh
evidenceloop run --input data/pipeline/redocred/documents.jsonl --dataset redocred --config configs/redocred.yaml --pipeline-variant proposer_critic_once --proposer-workers 4 --output-run-dir runs/redocred-once
```

完整流水线要求单个 Proposer worker；仅初始 Critic 变体支持有界预取。`critic_prompt_batch_size` 控制 GPU 同时处理的提示数量，与每个提示内的候选数量独立。

## 恢复与离线续接

新输出目录必须不存在。恢复显式使用 `--resume-run-dir`，成功文档跳过，失败文档重试；文档顺序、内容和运行参数不一致会报错。具体见[迁移说明](migration.md)。

```sh
evidenceloop run --input examples/documents.jsonl --resume-run-dir runs/example
evidenceloop offline critic --source-run-dir runs/direct --dataset rebel_sub --config configs/rebel_sub.yaml --output-run-dir runs/critic1
evidenceloop offline integrator --source-run-dir runs/critic1 --output-run-dir runs/integrator
evidenceloop offline final-critic --source-run-dir runs/integrator --output-run-dir runs/critic2
```

离线表示复用前一阶段结果；其中 Integrator 仍调用 API。可加 `--dry-run` 先验证源运行。批量发现使用 `--input-root`、`--output-root`、`--datasets`，会筛选已完成的正式规模运行；小样本用显式 `--source-run-dir`。

## 独立评测和报告

精确匹配不需要模型：

```sh
evidenceloop evaluate exact --gold examples/gold_triples.jsonl --predictions runs/example/kg.jsonl --dataset rebel_sub --output runs/example/exact_metrics.json
```

GraphJudge 和 CS/FS 需要额外模型或 API，命令详见[复现实验指南](reproduction.md)。BioRED 使用无向实体对，SciERC 只对两个指定关系使用对称规则。报告不会把未评测指标填成零。

重构前后的入口对照、历史断点迁移、论文与实现差异，以及本地实际验证结果，分别记录在 [migration.md](migration.md)、[implementation-differences.md](implementation-differences.md)、[validation.md](validation.md)。本次最终续作未新增或修改测试，也没有重跑论文实验。
