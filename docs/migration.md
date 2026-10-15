# Historical runs and entry-point migration

Original code, data, weights, logs and experiments stay in the old project. The new package does not provide old-script wrappers. Old module names below are a navigation table, not import aliases.

## Read and continue completed results

A finalized, successful historical run with valid state/progress and per-document checkpoints can be read without changing it. Explicit `offline ... --source-run-dir OLD_RUN --dry-run` validates the ordered records, document content and triples. The next actual stage always needs a different `--output-run-dir`. Reports accept existing saved summaries/metrics; give their output paths outside the historical run if preserving it read-only.

The reader supports legacy direct/pipeline/offline version-1 runs, the retained isolated related-baseline schema, and new version-2 runs. Bare KG exports can be evaluated with explicit predictions/gold paths but cannot by themselves supply verified resumable checkpoints. Completed results from incompatible model runtimes remain analyzable even when checkpoint migration is refused.

## Migrate a compatible interrupted run

`migrate-run` copies compatible **version-1 direct-Critic** pipeline, initial-Critic and final-Critic checkpoints into a new schema-2 directory. vLLM and unrelated runtime/checkpoint formats are rejected. No GPU/API initialization is needed to validate and copy.

```sh
evidenceloop migrate-run --source-run-dir archive/old-run --output-run-dir runs/migrated --input data/selected-documents.jsonl --dataset redocred --path-map /old/workspace=/current/workspace --path-map /old/models=models
```

The `OLD=NEW` paths are placeholders; supply your actual mapping. Absolute historical paths require explicit mappings, with the longest prefix applied first. The original directory is never rewritten. Keep mapped source files available. The canonical input must contain exactly the selected historical documents in the same order, including texts; do not pass the entire dataset when the old run selected a subset. The old input is read too because its original state may fingerprint only IDs. Offline migrations also verify the source-record fingerprint.

The tool checks dataset, document identities/content, base model, adapter, device, direct backend, BF16, safe merge, attention implementation, no-thinking setting and saved generation/grouping parameters. A supplied `--config` must match the inferred historical configuration. Failed documents remain failed and are retried on resume; successful documents are skipped. Available proposal caches are copied and validated, including external legacy caches mapped explicitly. A mismatch fails before creating the destination.

Resume a migrated pipeline using the same document file:

```sh
evidenceloop run --input data/selected-documents.jsonl --resume-run-dir runs/migrated
```

For an initial/final offline Critic migration, use the corresponding `offline critic` or `offline final-critic --resume-run-dir`. Runtime endpoints must match the recorded values. Credentials are supplied privately, never copied into new run state. Migration creates exports/progress; the resumed runner produces its normal summary.

New runs use exclusive output directories. Resume validates the stored configuration and content hashes rather than accepting different inputs. A stale `.run.lock` after forced termination must only be removed once the original process is confirmed stopped.

## Identifier and configuration changes

- `re_docred` is read as `redocred`; historical directory names remain readable.
- `pointwise` is read as `fixed1`; reports also locate old `pointwise/` variant directories.
- Candidate strategy identifiers are `fixed1`, `fixed2`, `fixed4`, `variable`. The old default `max4` grouping is retained internally and recorded separately from its canonical `fixed4` identifier.
- New configuration fields use `snake_case`; `PipelineConfig.from_file()` accepts the YAML templates under `configs/`.
- Personal paths/default API endpoints are removed. Generic `OPENAI_BASE_URL`/`OPENAI_API_KEY` are preferred; SIFLOW environment names are still readable.
- The core inference API receives documents only. Gold remains an independent evaluation input.

## Old-to-new commands

| Original entry | Installed command |
| --- | --- |
| `main.py` | `evidenceloop run --input ...` |
| `experiment/run_graphjudge_{rebel,redocred,biored,scierc}_10.py` | `run --dataset {rebel_sub,redocred,biored,scierc} --input ...` |
| `experiment/run_gpt_baselines.py` | `baseline direct --model ... --input ...` |
| `experiment/run_deepseek_baseline.py` | `baseline deepseek --input ...` |
| `experiment/run_gpt_proposer_critic_once.py` | `offline critic` |
| `experiment/run_offline_integrator.py` | `offline integrator` |
| `experiment/run_offline_final_critic.py` | `offline final-critic` |
| `experiment/build_redocred_proposal_cache.py` | `data proposal-cache` |
| `scripts/build_critic_sft_dataset.py` | `data legacy-phase1` |
| `scripts/build_critic_sft_dataset_phase2.py` | `data rebel-sft` |
| `scripts/build_{redocred,biored,scierc}_critic_sft_dataset.py` | `data {redocred,biored,scierc}-sft` |
| `scripts/build_redocred_critic_sft_ablation.py` | `data candidate-sets` |
| `scripts/build_rebel_phase2_pipeline_dataset.py` | `data rebel-documents` |
| `scripts/build_{redocred,biored,scierc}_pipeline_dataset.py` | `data {redocred,biored,scierc}-documents` |
| `training/train_critic_sft.py` | `critic train` |
| `training/evaluate_critic_sft.py` | `critic evaluate` |
| `training/check_critic_direct.py` | `critic check` |
| `experiment/prepare_related_baseline_models.py` | `baseline prepare`, `baseline train` |
| `experiment/run_related_baselines.py` | `baseline run` |
| `experiment/evaluate_related_baselines.py` | `evaluate baselines` |
| `experiment/evaluate_gpt_proposer_critic_once.py` | `evaluate critic` |
| `experiment/evaluate_deepseek_ablation_stages.py` | `evaluate stages` |
| `experiment/evaluate_redocred_candidate_set_ablation.py` | `evaluate candidate-sets` |
| `eval/evaluate_graphjudge_metrics.py` | `evaluate graphjudge` |
| `eval/evaluate_cs_fs.py` | `evaluate cs-fs` |
| Exact metrics formerly coupled to experiment inference | `evaluate exact` |
| `experiment/compare_graphjudge_runs.py` | `report runs` |
| `experiment/compare_pipeline_variants.py` | `report variants` |
| `experiment/report_redocred_critic_efficiency.py` | `report efficiency` |
| `experiment/summarize_inference_costs.py` | `report costs` |
| `scripts/report_critic_sft_statistics.py` | `report sft` |
| `scripts/report_critic_sft_phase2_statistics.py` | `report phase2` |
| `scripts/plot_critic_sft_statistics.py` | `report plot-sft` |
| Root plotting scripts with embedded historical measurements | `report plot --kind tradeoff` / `--kind distribution` with supplied JSON |
| Historical checkpoints formerly resumed in place | `migrate-run`, then explicit resume |

CLI options are intentionally explicit where old scripts searched personal directories or automatically read gold. Use each command's help; old options are not blindly forwarded.

## Module map

The full module map is also available as `module_map.json`. `docs/source_manifest.json` records hashes of original source files at migration time; these hashes identify the source baseline, not byte identity with refactored files. Removed unused file utilities have no replacement wrapper.

| Original module | Package module |
| --- | --- |
| `config.data.dataclass` | `evidenceloop.types` |
| `agent.base_agent` | `evidenceloop.agents.base` |
| `agent.proposer_agent` | `evidenceloop.agents.proposer` |
| `agent.critic_agent` | `evidenceloop.agents.critic` |
| `agent.integrator_agent` | `evidenceloop.agents.integrator` |
| `agent.kg_pipeline_controller` | `evidenceloop.pipeline.core` |
| `model.factory` | `evidenceloop.runtime.factory` |
| `model.direct_critic` | `evidenceloop.runtime.direct_critic` |
| `utils.config_handler` | `evidenceloop.settings` |
| `utils.path_tool` | `evidenceloop.paths` |
| `utils.logger_handler` | `evidenceloop.logging` |
| `utils.prompt_loader` | `evidenceloop.resources.prompt_loader` |
| `utils.document_loader` | `evidenceloop.data.documents` |
| `utils.triple_validation` | `evidenceloop.validation` |
| `utils.critic_runtime` | `evidenceloop.runtime.settings` |
| `utils.biored` | `evidenceloop.data.biored` |
| `utils.scierc` | `evidenceloop.data.scierc` |
| `utils.redocred` | `evidenceloop.data.redocred` |
| `utils.critic_sft` | `evidenceloop.data.sft` |
| `utils.critic_sft_statistics` | `evidenceloop.reports.sft_statistics` |
| `utils.critic_sft_phase2_statistics` | `evidenceloop.reports.phase2_statistics` |
| `experiment.run_graphjudge_rebel_10` | `evidenceloop.pipeline.run` |
| `experiment.run_graphjudge_redocred_10` | `evidenceloop.pipeline.run` |
| `experiment.run_graphjudge_biored_10` | `evidenceloop.pipeline.run` |
| `experiment.run_graphjudge_scierc_10` | `evidenceloop.pipeline.run` |
| `experiment.offline_stage_common` | `evidenceloop.experiments.history` |
| `experiment.run_gpt_baselines` | `evidenceloop.baselines.extract` |
| `experiment.run_deepseek_baseline` | `evidenceloop.baselines.extract` |
| `experiment.run_gpt_proposer_critic_once` | `evidenceloop.experiments.stages` |
| `experiment.run_offline_integrator` | `evidenceloop.experiments.stages` |
| `experiment.run_offline_final_critic` | `evidenceloop.experiments.stages` |
| `experiment.run_related_baselines` | `evidenceloop.baselines.run` |
| `experiment.prepare_related_baseline_models` | `evidenceloop.baselines.prepare` |
| `experiment.evaluate_related_baselines` | `evidenceloop.evaluation.baselines` |
| `experiment.evaluate_gpt_proposer_critic_once` | `evidenceloop.evaluation.critic` |
| `experiment.evaluate_deepseek_ablation_stages` | `evidenceloop.evaluation.stages` |
| `experiment.evaluate_redocred_candidate_set_ablation` | `evidenceloop.evaluation.candidate_sets` |
| `experiment.compare_graphjudge_runs` | `evidenceloop.reports.runs` |
| `experiment.compare_pipeline_variants` | `evidenceloop.reports.variants` |
| `experiment.build_redocred_proposal_cache` | `evidenceloop.experiments.proposal_cache` |
| `experiment.report_redocred_critic_efficiency` | `evidenceloop.reports.efficiency` |
| `experiment.summarize_inference_costs` | `evidenceloop.reports.costs` |
| `eval.evaluate_graphjudge_metrics` | `evidenceloop.evaluation.graphjudge` |
| `eval.evaluate_cs_fs` | `evidenceloop.evaluation.cs_fs` |
| `scripts.build_critic_sft_dataset` | `evidenceloop.data.builders.legacy_phase1` |
| `scripts.build_critic_sft_dataset_phase2` | `evidenceloop.data.builders.rebel_sft` |
| `scripts.build_redocred_critic_sft_dataset` | `evidenceloop.data.builders.redocred_sft` |
| `scripts.build_biored_critic_sft_dataset` | `evidenceloop.data.builders.biored_sft` |
| `scripts.build_scierc_critic_sft_dataset` | `evidenceloop.data.builders.scierc_sft` |
| `scripts.build_redocred_critic_sft_ablation` | `evidenceloop.data.builders.candidate_sets` |
| `scripts.build_rebel_phase2_pipeline_dataset` | `evidenceloop.data.builders.rebel_documents` |
| `scripts.build_redocred_pipeline_dataset` | `evidenceloop.data.builders.redocred_documents` |
| `scripts.build_biored_pipeline_dataset` | `evidenceloop.data.builders.biored_documents` |
| `scripts.build_scierc_pipeline_dataset` | `evidenceloop.data.builders.scierc_documents` |
| `scripts.report_critic_sft_statistics` | `evidenceloop.reports.sft` |
| `scripts.report_critic_sft_phase2_statistics` | `evidenceloop.reports.phase2` |
| `scripts.plot_critic_sft_statistics` | `evidenceloop.reports.plot_sft` |
| `training.train_critic_sft` | `evidenceloop.training.train` |
| `training.evaluate_critic_sft` | `evidenceloop.training.evaluate` |
| `training.check_critic_direct` | `evidenceloop.runtime.check` |
| `experiment.related_baselines` | `evidenceloop.baselines.related` |
| `experiment.related_baselines.adapters` | `evidenceloop.baselines.related.adapters` |
| `experiment.related_baselines.common` | `evidenceloop.baselines.related.common` |
| `experiment.related_baselines.prompts` | `evidenceloop.baselines.related.prompts` |
| `experiment.related_baselines.runtime` | `evidenceloop.baselines.related.runtime` |
| `experiment.related_baselines.training` | `evidenceloop.baselines.related.training` |
