import json
import time
from typing import Dict, List

from evidenceloop.agents.base import BaseAgent
from evidenceloop.logging import logger
from evidenceloop.resources.prompt_loader import (
    load_critic_call_prompt_batch,
    load_critic_system_prompt,
)
from evidenceloop.types import Triple


class CriticAgent(BaseAgent):
    def __init__(
        self,
        name: str,
        role: str,
        *,
        llm=None,
        runtime_settings=None,
    ):
        self._llm_override = llm
        if runtime_settings is None:
            from evidenceloop.runtime.settings import resolve_critic_runtime

            runtime_settings = resolve_critic_runtime({})
        self.runtime_settings = dict(runtime_settings)
        self.max_batch_size = int(self.runtime_settings["batch_size"])
        self.prompt_batch_size = int(self.runtime_settings.get("prompt_batch_size", 1))
        super().__init__(name=name, role=role)

    def _get_system_prompt(self) -> str:
        return load_critic_system_prompt()

    def _init_llm(self):
        if self._llm_override is not None:
            return self._llm_override
        from evidenceloop.runtime.direct_critic import DirectCriticEngine

        return DirectCriticEngine(self.runtime_settings)

    def validate_runtime(self) -> Dict:
        if not hasattr(self.llm, "validate_runtime"):
            raise RuntimeError("Critic model does not support direct runtime validation")
        return self.llm.validate_runtime()

    def _public_triple_dict(self, triple: Triple) -> Dict:
        return triple.to_spo_dict()

    def _format_single_triple_info(self, idx: int, triple: Triple, origin: str) -> Dict:
        return {
            "triple_idx": idx,
            "origin": origin,
            "triple": self._public_triple_dict(triple),
        }

    def _extract_json_obj(self, response: str) -> Dict:
        raw = self.extract_json(response or "")
        if not raw:
            return {}
        try:
            return json.loads(self.fix_unescaped_quotes(raw))
        except json.JSONDecodeError:
            logger.warning(f"Critic JSON parse failed: {raw[:200]}...")
            return {}

    def _review_chunks(
        self,
        triples: List[Triple],
        stage: str,
        origins: List[str],
        chunk_sizes: List[int] = None,
    ) -> List[Dict]:
        chunks = []
        if chunk_sizes is None:
            ranges = [
                (start, min(start + self.max_batch_size, len(triples)))
                for start in range(0, len(triples), self.max_batch_size)
            ]
        else:
            ranges = []
            cursor = 0
            for size in chunk_sizes:
                if not isinstance(size, int) or not 1 <= size <= self.max_batch_size:
                    raise ValueError(
                        "Critic chunk sizes must be positive integers no greater than max_batch_size"
                    )
                end = cursor + size
                if end > len(triples):
                    raise ValueError("Critic chunk sizes exceed the triple count")
                ranges.append((cursor, end))
                cursor = end
            if cursor != len(triples):
                raise ValueError("Critic chunk sizes must cover every triple exactly once")

        for start, end in ranges:
            chunk = triples[start:end]
            triples_info = [
                self._format_single_triple_info(
                    local_index,
                    triple,
                    origins[start + local_index]
                    if start + local_index < len(origins)
                    else "proposer",
                )
                for local_index, triple in enumerate(chunk)
            ]
            chunks.append(
                {
                    "start": start,
                    "end": end,
                    "prompt": load_critic_call_prompt_batch().format(
                        review_stage=stage,
                        source_text=chunk[0].source_text,
                        triples_info=json.dumps(triples_info, ensure_ascii=False, indent=2),
                    ),
                }
            )
        return chunks

    def _merge_chunk_results(
        self,
        chunks: List[Dict],
        responses: List,
    ) -> Dict:
        merged_results = []
        raw_chunks = []
        errors = []
        for chunk, generated in zip(chunks, responses):
            start = chunk["start"]
            end = chunk["end"]
            response = str(getattr(generated, "content", generated) or "")
            generation_error = str(getattr(generated, "error", "") or "")
            raw_chunks.append(
                {
                    "start_idx": start,
                    "end_idx_exclusive": end,
                    "response": response,
                }
            )
            if generation_error:
                errors.append(f"indices {start}-{end - 1}: {generation_error}")
                continue
            data = self._extract_json_obj(response)
            results = data.get("results", []) if data else []
            if not data:
                errors.append(
                    f"indices {start}-{end - 1}: Critic response did not contain valid JSON"
                )
                continue
            if not isinstance(results, list):
                errors.append(f"indices {start}-{end - 1}: Critic results must be a list")
                continue
            for result in results:
                if not isinstance(result, dict):
                    continue
                local_index = result.get("triple_idx")
                chunk_size = end - start
                if not isinstance(local_index, int) or not 0 <= local_index < chunk_size:
                    errors.append(
                        f"indices {start}-{end - 1}: invalid local triple_idx {local_index!r}"
                    )
                    continue
                mapped = dict(result)
                mapped["triple_idx"] = start + local_index
                merged_results.append(mapped)

        raw_response = (
            (
                raw_chunks[0]["response"]
                if len(raw_chunks) == 1
                else json.dumps(raw_chunks, ensure_ascii=False, separators=(",", ":"))
            )
            if raw_chunks
            else ""
        )
        return {
            "results": merged_results,
            "raw_response": raw_response,
            "parse_error": "; ".join(errors),
        }

    def review_documents_batch(self, requests: List[Dict]) -> List[Dict]:
        prepared = []
        all_chunks = []
        for document_index, request in enumerate(requests):
            triples = list(request.get("triples") or [])
            stage = str(request.get("stage") or "")
            if stage not in {"initial_review", "final_validation"}:
                raise ValueError(f"unsupported Critic stage: {stage!r}")
            origins = list(request.get("origins") or ["proposer"] * len(triples))
            requested_sizes = request.get("chunk_sizes")
            chunks = (
                self._review_chunks(
                    triples,
                    stage,
                    origins,
                    list(requested_sizes) if requested_sizes is not None else None,
                )
                if triples
                else []
            )
            prepared.append(chunks)
            all_chunks.extend({**chunk, "document_index": document_index} for chunk in chunks)

        started = time.perf_counter()
        if hasattr(getattr(self, "llm", None), "generate_batch") and all_chunks:
            messages = [
                [("system", self.system_prompt), ("user", chunk["prompt"])] for chunk in all_chunks
            ]
            all_responses = self.llm.generate_batch(
                messages,
                batch_size=self.prompt_batch_size,
            )
        else:
            all_responses = [self.call_llm(chunk["prompt"]) for chunk in all_chunks]
        elapsed = time.perf_counter() - started
        if len(all_responses) != len(all_chunks):
            raise RuntimeError("Critic batch output count does not match the prepared prompt count")

        responses_by_document = [[] for _ in prepared]
        for chunk, response in zip(all_chunks, all_responses):
            responses_by_document[chunk["document_index"]].append(response)
        prompt_count = len(all_chunks)
        output = []
        for chunks, responses in zip(prepared, responses_by_document):
            elapsed_share = elapsed * len(chunks) / prompt_count if prompt_count else 0.0
            output.append(
                {
                    "payload": self._merge_chunk_results(chunks, responses),
                    "elapsed_seconds": round(elapsed_share, 6),
                    "prompt_count": len(chunks),
                }
            )
        return output

    def review_triples_batch(
        self,
        triples: List[Triple],
        stage: str,
        origins: List[str] = None,
        chunk_sizes: List[int] = None,
    ) -> Dict:
        n = len(triples)
        batch_size = int(getattr(self, "max_batch_size", 4))
        if not 1 <= batch_size <= 4:
            raise ValueError("Critic max_batch_size must be between 1 and 4")
        self.max_batch_size = batch_size
        return self.review_documents_batch(
            [
                {
                    "triples": triples,
                    "stage": stage,
                    "origins": list(origins or ["proposer"] * n),
                    "chunk_sizes": chunk_sizes,
                }
            ]
        )[0]["payload"]
