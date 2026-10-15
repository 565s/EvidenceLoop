# Paper and implementation differences

This is a record of existing behavior, not a change to the method. The supplied EvidenceLoop manuscript was used as background. It is not redistributed here, and publication/author/repository identifiers have not been invented. Research changes should be evaluated separately from this restructuring.

| Topic | Manuscript description or comparison concern | Preserved implementation |
| --- | --- | --- |
| Evidence | Section 3.6 describes normalized whitespace, remapping and endpoint checks | Exact `source_text.find(evidence)`, first occurrence, Python character offsets; no new endpoint-in-evidence validation |
| Deduplication | Evidence/priority-aware selection can suggest a preference among duplicates | Stable first occurrence after lowercasing and folding whitespace in the SPO key |
| Critic actions | KEEP, DELETE, FIX, REWRITE | All four preserved; unknown/missing/illegal reviews take the existing fallback |
| Failed review | Invalid output may suggest universal rejection | Proposer-origin input survives fallback; unverified Integrator-origin final additions are deleted |
| JSON parsing | More robust structured decoding could change outcomes | Greedy object extraction and the original unescaped-quote repair remain where originally used; direct extraction and third-party parsers remain separate |
| GraphJudge matching | Paper equations can be read as independent reusable matches | Official-style Hungarian one-to-one matching is the default; `--matching-mode reusable` is explicit |
| CS | Generic similarity or one-to-one alignment would give different numbers | Normalize each SPO element embedding, sum and normalize the triple vector, reusable best prediction per gold; default threshold 0.95 |
| FS | A reference-based formulation would differ | Existing API judge uses the source document as its factual reference; original API output parsing remains |
| CS/FS backend | Local/reference GenRES execution would use different models | Configured OpenAI-compatible embeddings/chat API; this is not claimed numerically identical to every GenRES backend |
| Direction rules | A uniform directed graph score would affect BioRED/SciERC | BioRED is undirected; SciERC is symmetric only for `is compared with` and `is conjoined with`; others directed |
| Exact normalization | Different historical evaluators have distinct normalization | Pipeline exact matching keeps strip/lower keys; third-party evaluation retains its original whitespace/case folding conventions |
| Offline Integrator | The initial extraction model may differ between baselines | The historical offline stage independently uses DeepSeek, temperature 0.5, 2048 max tokens, 120-second timeout and no thinking unless explicitly configured |
| Candidate sets | GPU batch size and candidates per prompt can be confused | They are independent. Fixed groups retain the tail; variable groups use seeded per-document randomness |
| Dataset selection | Paper counts alone do not specify eligibility/filtering | Original builder filtering, seeds and manifests remain authoritative; compare generated metadata and document fingerprints |
| Additional ablations | Context interference, random ordering and evidence-removal experiments may be discussed in the manuscript | No standalone implementation was found among the migrated original entry points; no replacement experiment or result was fabricated |

## Engineering changes affecting how runs are managed

Import-time YAML/client/controller creation is replaced by explicit configuration and lazy per-Pipeline factories. Batch output is separate from single-document processing. New run schema, content hashes, source validation, exclusive directories and explicit resume replace truncation-prone startup. Old identifiers remain read aliases. Provider-specific endpoint defaults and personal absolute paths are removed; supply your own configuration.

Input texts, prompts, triple order, candidate generation, evidence lookup, origin labels and error/fallback logic remain subject to the migrated regression checks. Exact timing, API outputs and GPU numerics are not guaranteed by offline checks. The final continuation adds documentation, packaging and production fixes while leaving existing test files unchanged. No paid requests, downloads, training or complete paper reruns were performed for this delivery.
