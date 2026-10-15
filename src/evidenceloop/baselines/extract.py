"""Document-only direct extraction with the historical baseline response parser."""

from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path

from evidenceloop import Pipeline, PipelineConfig
from evidenceloop.baselines.parsing import parse_gpt_response
from evidenceloop.data.documents import load_documents
from evidenceloop.io import load_dotenv_file, read_json_object
from evidenceloop.pipeline.run import RunManager
from evidenceloop.resources.prompt_loader import (
    load_proposer_extract_call_prompt,
    load_proposer_system_prompt,
)
from evidenceloop.runtime.factory import LazyRuntime


class DirectExtractor:
    def __init__(self, config, llm=None):
        self.config = config
        self.llm = llm if llm is not None else LazyRuntime(self._create_client)

    def _create_client(self):
        from langchain_openai import ChatOpenAI

        endpoint = (
            os.environ.get(self.config.base_url_env)
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("SIFLOW_BASE_URL")
        )
        key = (
            os.environ.get(self.config.api_key_env)
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("SIFLOW_API_KEY")
        )
        if not endpoint or not key:
            raise ValueError("Direct extraction requires an explicit API endpoint and key")
        options = {}
        if "deepseek" in self.config.proposer_model_name.lower():
            options["extra_body"] = {"enable_thinking": False}
        return ChatOpenAI(
            model=self.config.proposer_model_name,
            base_url=endpoint,
            api_key=key,
            temperature=self.config.proposer_temperature,
            max_tokens=2048,
            max_retries=0,
            **options,
        )

    def propose_triples(self, document_id, text):
        response = self.llm.invoke(
            [
                ("system", load_proposer_system_prompt()),
                ("user", load_proposer_extract_call_prompt().format(text=text)),
            ]
        )
        content = getattr(response, "content", response)
        return parse_gpt_response(str(content or "").strip(), document_id, text)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="evidenceloop baseline direct")
    parser.add_argument("--input", required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--model")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--config")
    parser.add_argument("--output-run-dir")
    parser.add_argument("--resume-run-dir")
    parser.add_argument("--proposer-workers", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--env-file")
    args = parser.parse_args(argv)
    if args.env_file:
        load_dotenv_file(Path(args.env_file))
    saved = None
    if args.resume_run_dir:
        saved = read_json_object(Path(args.resume_run_dir) / "run_state.json")
        if saved.get("schema_version") != 2 or saved.get("kind") != "direct":
            raise ValueError("Resume requires a version-2 direct-extraction run")
    config = (
        PipelineConfig.from_file(args.config)
        if args.config
        else PipelineConfig(**saved["runtime"]["config"])
        if saved
        else PipelineConfig(pipeline_variant="proposer_critic_once", proposer_temperature=0.0)
    )
    overrides = {"pipeline_variant": "proposer_critic_once"}
    if args.model is not None:
        overrides["proposer_model_name"] = args.model
    if args.temperature is not None:
        overrides["proposer_temperature"] = args.temperature
    config = replace(config, **overrides)
    runtime = saved["runtime"] if saved else {}
    dataset = args.dataset or runtime.get("dataset", "rebel_sub")
    workers = (
        args.proposer_workers
        if args.proposer_workers is not None
        else runtime.get("proposer_workers", 1)
    )
    documents = load_documents(args.input)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit must be positive")
        documents = documents[: args.limit]
    pipeline = Pipeline(config=config, proposer_agent=DirectExtractor(config))
    result = RunManager(pipeline, dataset=dataset, stage="direct", proposer_workers=workers).run(
        documents,
        args.output_run_dir,
        resume_run_dir=args.resume_run_dir,
    )
    print(result)
    return int(read_json_object(result / "progress.json")["failed"] > 0)
