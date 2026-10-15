# EvidenceLoop

Evidence-grounded knowledge graph extraction from complete documents, with Proposer extraction, a local Critic, an Integrator coverage audit, and final Critic validation.

This repository reorganizes the research implementation for installation and review. Algorithm behavior, prompt text, response parsing and failure policies are preserved. See [implementation differences](docs/implementation-differences.md) before comparing results with the paper. Model weights, full datasets and historical runs are supplied separately by the user. Publication metadata and model download locations have not been filled in because they were not provided.

## Install

Python 3.10 or later is required. In your own virtual environment:

```sh
python -m pip install -e .
evidenceloop --help
python -m evidenceloop --help
```

Choose extras for the work you intend to run:

| Extra | Purpose |
| --- | --- |
| `critic` | Local BF16 Qwen3-8B with a safely merged LoRA adapter |
| `train` | Critic LoRA training and dataset tokenization |
| `eval` | GraphJudge soft metrics and optional evaluation dependencies |
| `plot` | Report figures |
| `dev` | Existing offline checks, formatting and package builds |

For example, `python -m pip install -e ".[critic]"`. Install a PyTorch build appropriate for your GPU first. The direct Critic requires CUDA and BF16 support; CPU execution is not a substitute for this runtime. The five third-party baselines use [separate environments](docs/reproduction.md#third-party-baselines).

Installation, import and command help do not initialize API clients or load models. Actual API stages require an explicitly configured OpenAI-compatible endpoint and key. Copy `.env.example` to a private file and use `--env-file` on commands that expose it, or export the variables in your shell. No endpoint or credential is supplied by default.

## Documents and inference

Canonical input is JSONL or a JSON array:

```json
{"document_id": "example-1", "text": "Ada Lovelace was born in London."}
```

Each record is processed as a complete document, without question text or automatic chunking. Titles are ignored. Historical ID/text aliases and `.source` inputs remain readable; gold triples are separate evaluation inputs.

After placing your base model and the dataset-specific adapter at the paths in a configuration:

```sh
evidenceloop data validate examples/documents.jsonl
evidenceloop run --input examples/documents.jsonl --config configs/rebel_sub.yaml --output-run-dir runs/example
```

The second command performs real model work. It is not an offline demonstration. All relative input, model and output paths are relative to your current working directory. Use absolute paths when running from elsewhere. The output directory must be new. Recover a compatible run using the same input and `--resume-run-dir runs/example`.

```python
from evidenceloop import Document, Pipeline, PipelineConfig

pipeline = Pipeline(config=PipelineConfig.from_file("configs/rebel_sub.yaml"))
record = pipeline.process(Document("example-1", "Ada Lovelace was born in London."))
triples = record.final_triples
```

`Pipeline`, `PipelineConfig`, `Document`, `Triple` and `DocumentPipelineRecord` are public interfaces. Agents can also be injected into `Pipeline`. `process()` returns one record; batch output, concurrency and recovery belong to `RunManager`, described in [architecture](docs/architecture.md).

## Commands

Every entry below also works through `python -m evidenceloop`. Use the specific command's `--help` for options.

| Group | Actions |
| --- | --- |
| `run` | Full pipeline or `--pipeline-variant proposer_critic_once` |
| `offline` | `critic`, `integrator`, `final-critic` on completed candidate runs |
| `data` | Four dataset SFT/document builders, `candidate-sets`, `proposal-cache`, `legacy-phase1`, `validate` |
| `critic` | `train`, `evaluate`, `check` |
| `baseline` | `direct`/`deepseek`, `prepare`, `train`, `run` |
| `evaluate` | `exact`, `graphjudge`, `cs-fs`, `baselines`, `critic`, `stages`, `candidate-sets` |
| `report` | `runs`, `variants`, `efficiency`, `costs`, `sft`, `phase2`, `plot`, `plot-sft` |
| `migrate-run` | Validate and copy compatible historical direct-Critic checkpoints |

Dataset identifiers are `rebel_sub`, `redocred`, `biored`, `scierc`; `re_docred` remains a read alias. Candidate strategies are `fixed1`, `fixed2`, `fixed4`, `variable`; `pointwise` remains a read alias. The default uses the original maximum-four grouping; see the fixed-four/default distinction in the reproduction guide.

## Documentation

- [中文入门](docs/quickstart.zh-CN.md)
- [Architecture and data contracts](docs/architecture.md)
- [Reproduction commands and baseline environments](docs/reproduction.md)
- [Historical migration and old-to-new entry points](docs/migration.md)
- [Paper/implementation differences](docs/implementation-differences.md)
- [Offline delivery verification and remaining validation](docs/validation.md)

## Development and licensing

```sh
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check src scripts
python -m ruff format --check src scripts
python -m build
python scripts/audit_release.py
```

The existing regression suite includes frozen reference implementations under `tests/reference`; these are not installed as application modules. No additional tests were added during the final continuation, as requested. Windows and Linux CI is configured under `.github/workflows/ci.yml`.

Project code is offered under [MIT](LICENSE). Third-party prompts, relation metadata, packages, models and datasets retain their own terms; see [third-party notices](THIRD_PARTY_NOTICES.md). This local delivery has not been published to GitHub.
