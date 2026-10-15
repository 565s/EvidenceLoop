# Isolated related-work baselines

Adapters implement RAKG, iText2KG, PiVe, KGGen and GraphJudge using the recorded upstream behavior. Install each method's `requirements/METHOD.txt` in a separate environment; preserve upstream version constraints. Install EvidenceLoop there with `pip install -e . --no-deps`. Use `evidenceloop baseline run --backend-python METHOD=PYTHON` from the main environment to dispatch.

Commands are `baseline prepare` (check-only by default; explicit `--download` for weights), `baseline run --preflight`, `baseline run`, `baseline train --train-pive-data DIR` (PiVe), and independent `evaluate baselines`. Use the command help for explicit local paths and runtime settings. Inference reads documents only.

Prompt resources and the RAKG MIT notice are installed with the package. Full reproduction guidance and third-party notices are at the repository root under `docs/reproduction.md` and `THIRD_PARTY_NOTICES.md`. No real third-party training or inference was performed as part of the offline refactor acceptance.
