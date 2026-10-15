# Reproduction guide

Commands in this guide are recipes. Replace example paths with your files. Model/API commands have not been executed for delivery. Use a fresh directory for each run and retain configuration, input manifests, dataset metadata, adapter identity and evaluation settings. Do not compare scores across different document cohorts or silently substitute adapters.

## Dataset preparation

The four canonical identifiers are `rebel_sub`, `redocred`, `biored`, `scierc`. Source archives, annotations and splits must be acquired under their original terms; this repository contains relation mappings and two manually written example documents only.

| Dataset | SFT builder | Canonical document builder | Formal cohort preset |
| --- | --- | --- | --- |
| REBEL-Sub | `data rebel-sft` | `data rebel-documents` | 1000 |
| Re-DocRED | `data redocred-sft` | `data redocred-documents` | 500 |
| BioRED | `data biored-sft` | `data biored-documents` | 100 |
| SciERC | `data scierc-sft` | `data scierc-documents` | 100 |

These cohort presets govern retained formal-run discovery/evaluation. `run --limit` and explicit offline sources support smaller experiments. Builders preserve their original eligibility criteria and seed behavior, so inspect their metadata rather than assuming raw dataset counts equal eligible document counts.

```sh
evidenceloop data rebel-sft --source-zip data/raw/rebel.zip --output-dir data/critic_sft/rebel_phase2
evidenceloop data rebel-documents --source-zip data/raw/rebel.zip --critic-test-file data/critic_sft/rebel_phase2/test.jsonl --output-dir data/pipeline/rebel_sub

evidenceloop data redocred-sft --train-file data/raw/redocred/train.json --validation-file data/raw/redocred/dev.json --test-file data/raw/redocred/test.json --relation-map src/evidenceloop/resources/config/docred_rel_info.json --output-dir data/critic_sft/redocred_phase2
evidenceloop data redocred-documents --test-file data/raw/redocred/test.json --relation-map src/evidenceloop/resources/config/docred_rel_info.json --output-dir data/pipeline/redocred

evidenceloop data biored-sft --source-zip data/raw/biored.zip --output-dir data/critic_sft/biored_phase2
evidenceloop data biored-documents --source-zip data/raw/biored.zip --output-dir data/pipeline/biored

evidenceloop data scierc-sft --train-file data/raw/scierc/train.jsonl --validation-file data/raw/scierc/dev.jsonl --test-file data/raw/scierc/test.jsonl --output-dir data/critic_sft/scierc_phase2
evidenceloop data scierc-documents --test-file data/raw/scierc/test.jsonl --output-dir data/pipeline/scierc

evidenceloop data validate data/pipeline/redocred/documents.jsonl
evidenceloop data validate data/critic_sft/redocred_phase2/train.jsonl --kind sft
```

The explicit relation-map paths above refer to a source checkout. Installed callers can locate the bundled map with `importlib.resources.files("evidenceloop.resources.config").joinpath("docred_rel_info.json")`, or supply their own map. Generated document directories contain `documents.jsonl` and separate `gold_triples.jsonl`; inference accepts only the document file.

`data legacy-phase1` preserves the earlier REBEL SFT builder. Its explicit command recipe is `configs/legacy_phase1.json`, with the original 20000/2000/2000 requested document counts and seed 42. It is a historical experiment, not the phase-two default.

## Critic training and evaluation

Install the `train` extra and prepare local model files. No weights are included. Train one adapter per dataset:

```sh
evidenceloop critic train --model-name-or-path models/Qwen3-8B --train-file data/critic_sft/redocred_phase2/train.jsonl --validation-file data/critic_sft/redocred_phase2/validation.jsonl --output-dir outputs/critic_qwen3_8b_lora_redocred_phase2 --max-steps 500
evidenceloop critic evaluate --model-name-or-path models/Qwen3-8B --adapter-path outputs/critic_qwen3_8b_lora_redocred_phase2/adapter --test-file data/critic_sft/redocred_phase2/test.jsonl --output-dir outputs/redocred_critic_evaluation
evidenceloop critic check --base-model models/Qwen3-8B --adapter-path outputs/critic_qwen3_8b_lora_redocred_phase2/adapter --test-file data/critic_sft/redocred_phase2/test.jsonl
```

Retained training defaults: BF16, LoRA rank 16/alpha 32/dropout 0.05, sequence length 4096, learning rate 2e-4, AdamW, cosine scheduling, batch size 1, gradient accumulation 16, seed 42 and 500 steps. `--smoke-test` is real model training for two steps, not an offline check. `--validate-only` validates the prepared training inputs. Independent Critic evaluation can consume `--predictions-file` without regenerating predictions.

Inference safely merges the adapter into one in-process BF16 model, with SDPA, no thinking and a maximum of four candidates per Critic request. Runtime sequence/token limits remain distinct from training limits; use the saved runtime configuration when comparing runs.

## Full, once and offline stages

```sh
evidenceloop run --input data/pipeline/redocred/documents.jsonl --dataset redocred --config configs/redocred.yaml --output-run-dir runs/redocred-full
evidenceloop run --input data/pipeline/redocred/documents.jsonl --dataset redocred --config configs/redocred.yaml --pipeline-variant proposer_critic_once --proposer-workers 4 --output-run-dir runs/redocred-once

evidenceloop baseline direct --input data/pipeline/redocred/documents.jsonl --dataset redocred --model YOUR_MODEL --output-run-dir runs/direct
evidenceloop offline critic --source-run-dir runs/direct --dataset redocred --config configs/redocred.yaml --output-run-dir runs/critic1
evidenceloop offline integrator --source-run-dir runs/critic1 --output-run-dir runs/integrator
evidenceloop offline final-critic --source-run-dir runs/integrator --output-run-dir runs/critic2
```

`baseline deepseek` uses the same direct extraction implementation and defaults to the configured DeepSeek model. The direct baseline keeps its original direct-response parser and temperature 0.0 default. Offline Integrator preserves its separate historical request options (2048 tokens, 120 seconds, no thinking, worker default 4). The final Critic inherits Critic1's exact base, adapter and device. Each stage writes to its own new directory and never reads gold.

For a completed historical source, use an explicit `--source-run-dir` and `--dry-run` first. To discover the latest complete formal sources per dataset/model, supply `--input-root ROOT --output-root NEW_ROOT --datasets redocred`; roots are explicit rather than personal server locations. Each selected run executes in a separate process. Resume uses `--resume-run-dir` and the saved source identity.

## Candidate-set ablation

```sh
evidenceloop data candidate-sets --train-file data/raw/redocred/train.json --validation-file data/raw/redocred/dev.json --test-file data/raw/redocred/test.json --relation-map src/evidenceloop/resources/config/docred_rel_info.json --output-root data/critic_sft/redocred_candidate_sets
evidenceloop data proposal-cache --input data/pipeline/redocred/documents.jsonl --config configs/redocred.yaml --output-dir runs/shared-proposals --workers 4
```

The builder creates paired `fixed1`, `fixed2`, `fixed4`, `variable` variants using one candidate manifest and eligible document pool. Train four independent adapters using `critic train --candidate-set-variant VARIANT` with the corresponding train/validation paths and distinct output directories. Supply each adapter through its own configuration when running:

```sh
evidenceloop run --input data/pipeline/redocred/documents.jsonl --dataset redocred --config configs/redocred.yaml --pipeline-variant proposer_critic_once --critic-candidate-set-size fixed4 --proposal-cache-dir runs/shared-proposals --require-proposal-cache --output-run-dir runs/redocred-fixed4
```

Repeat with the other three configurations/strategies. The cache checks exact document content/order, parser, model, temperature and endpoint. Never rebuild proposals separately for a purportedly paired comparison. `pointwise` is accepted when reading old `fixed1` metadata/directories.

Omitting the strategy retains the old `max4` grouping, which uses the configured Critic candidate batch size (at most four). Its public strategy identifier is `fixed4`, with `candidate_grouping` recorded separately. Explicit `fixed4` uses literal groups of four and retains a short final group; fixed1/fixed2 are literal sizes. The default batch size is four in both cases. Variable grouping is seeded per document. `critic_prompt_batch_size` controls GPU prompt batching independently.

`evaluate candidate-sets` retains the original candidate-set evaluation/aggregation, including an optional real order-consistency check. Inspect `--help` before use: real adapter inference is not part of offline acceptance. `report efficiency` reads existing run/cache statistics without loading gold or models.

## Independent metrics and reports

```sh
evidenceloop evaluate exact --gold data/pipeline/redocred/gold_triples.jsonl --predictions runs/redocred-full/kg.jsonl --dataset redocred --empty-gold-policy skip --output runs/redocred-full/exact_metrics.json
evidenceloop evaluate graphjudge --run-dir runs/redocred-full --gold-file data/pipeline/redocred/gold_triples.jsonl --pred-file runs/redocred-full/kg.jsonl --bert-model models/bert-large-uncased --empty-gold-policy skip
evidenceloop evaluate cs-fs --run-dir runs/redocred-full --gold-file data/pipeline/redocred/gold_triples.jsonl --documents-file data/pipeline/redocred/documents.jsonl --env-file .env
```

GraphJudge uses Hungarian matching by default. Specify `--entity-order-mode undirected` for BioRED. For SciERC, use `--entity-order-mode predicate_aware --symmetric-predicate "is compared with" --symmetric-predicate "is conjoined with"`. REBEL/Re-DocRED are directed. Re-DocRED soft evaluation skips empty gold; the exact command exposes this choice explicitly. Keep matching, empty-gold policy, BERT model/layers and cohort fingerprints with results. CS/FS makes embedding/chat API calls; its optional cache is not included in the release.

```sh
evidenceloop report runs --baseline-run-dir runs/base --lora-run-dir runs/adapted --output reports/comparison.json
evidenceloop report variants --help
evidenceloop report costs --run-dir runs/redocred-full --output reports/costs.md
evidenceloop report efficiency --help
evidenceloop report sft --help
evidenceloop report phase2 --help
evidenceloop report plot --help
evidenceloop report plot-sft --help
```

Run comparison reads an optional `exact_metrics.json` produced above; missing metrics stay null. Historical summaries containing exact metrics remain readable. Cost reports summarize observed runtime/token fields and disclose missing usage, partial timing and cache reuse. They do not reconstruct prices or claim unrecorded cost reductions. Plot inputs are supplied JSON rather than hardcoded historical paper numbers.

## Third-party baselines

Methods are `rakg`, `itext2kg`, `pive`, `kggen`, `graphjudge`. Their pinned dependencies are installed independently, under `src/evidenceloop/baselines/related/requirements/`. In each method's environment, install the corresponding requirements, then install this project using `pip install -e . --no-deps` so the main environment's LangChain constraints do not replace upstream pins. Add `PyYAML` and `python-dotenv` if the selected requirements do not provide them. Use `--backend-python METHOD=PYTHON` to dispatch from your main environment.

For example, iText2KG 0.0.7 needs its upstream LangChain 0.3.0 / langchain-openai 0.2.0 environment and HTTPX 0.27.2; the main package deliberately does not force these older constraints globally. Quantization and PiVe training have separate requirement files. See [third-party notices](../THIRD_PARTY_NOTICES.md) for the recorded upstream identities.

```sh
evidenceloop baseline prepare --method pive --dataset redocred --model-root models/related_baselines
evidenceloop baseline run --method pive --dataset redocred --data-dir data/pipeline/redocred --model-root models/related_baselines --preflight
```

Check-only preparation/preflight does not download weights. Downloads require explicit `baseline prepare --download`. PiVe data conversion uses `baseline prepare --convert-pive-data TRAIN_FILE`; training uses `baseline train --method pive --train-pive-data CONVERTED_DIR --training-output-dir outputs/pive`. These are genuine model operations and remain unexecuted. Other methods retain their original model preparation/run steps; do not imply they have PiVe's fine-tuning path.

`baseline run` reads documents only, keeps stage traces and retries failed documents. `evaluate baselines` is the separate gold-reading entry. Source requirements and adapters are preserved, but real third-party model validation requires their individual environments and artifacts.
