# Third-party notices

The MIT license in this repository applies to project code that the project owner can license. It does not relicense upstream packages, datasets, models or externally sourced resources. Author and publication metadata for EvidenceLoop have not been guessed.

The following upstream identities and versions are recorded in the original implementation. They have not been refreshed against the network during this offline refactor:

| Component | Recorded source | Version or revision |
| --- | --- | --- |
| RAKG | `https://github.com/KnowledgeXLab/RAKG` | `7cdd1d230be91ac9a2e6661ae9d60f1d28e7391f` |
| iText2KG | `https://github.com/AuvaLab/itext2kg` | package `itext2kg==0.0.7` |
| PiVe | `https://github.com/Jiuzhouh/PiVe` | `3c1ec7e864ea8fdcadee9e9170ede795df4a1d48` |
| KGGen | `https://github.com/stair-lab/kg-gen` | package `kg-gen==0.2.0` |
| GraphJudge | `https://github.com/hhy-huang/GraphJudge` | `0ae5913d49d8b530c254cbc0cefa71a57d107bb5` |

RAKG-derived prompt resources are under `src/evidenceloop/baselines/related/resources/rakg_prompts.json`. Their accompanying upstream MIT notice, including `Copyright (c) 2025 LMMApplication`, is preserved verbatim in `RAKG_LICENSE.txt` and included in installed distributions. Consult that notice for these resources.

The other baseline packages/source trees are not vendored. Their installation requirements retain known upstream version constraints under `baselines/related/requirements/`. Consult their upstream license files for use and redistribution; this refactor has not asserted an unverified license classification for them.

Relation metadata for Re-DocRED, BioRED and SciERC is packaged under `resources/config/` to preserve the original label mapping. Such mappings derive from the corresponding datasets and remain subject to their terms. Full datasets and weights are excluded. Qwen, LoRA adapters, BERTScore/BERT models, embedding models, PiVe/GraphJudge artifacts and API-provided models retain their own licenses or service terms. The original recorded model identifiers are configuration, not bundled artifacts or an endorsement of redistribution rights.

Python dependencies are installed separately; their upstream licenses remain applicable. This delivery does not bundle a Python environment. Publication links, author identities and custom adapter download URLs should be supplied by the project owner before publication.
