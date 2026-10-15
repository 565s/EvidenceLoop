"""Build the same verified proposal cache used by RunManager, without reading gold."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evidenceloop import Pipeline, PipelineConfig
from evidenceloop.data.documents import load_documents
from evidenceloop.io import load_dotenv_file
from evidenceloop.pipeline.run import RunManager


def main(argv=None):
    parser = argparse.ArgumentParser(prog="evidenceloop data proposal-cache")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--env-file")
    args = parser.parse_args(argv)
    if args.env_file:
        load_dotenv_file(Path(args.env_file))
    documents = load_documents(args.input)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit must be positive")
        documents = documents[: args.limit]
    if len({d.document_id for d in documents}) != len(documents):
        raise ValueError("Duplicate document IDs")
    if not documents:
        raise ValueError("Input contains no documents")
    config = (
        PipelineConfig.from_file(args.config)
        if args.config
        else PipelineConfig(pipeline_variant="proposer_critic_once")
    )
    from dataclasses import replace

    config = replace(config, pipeline_variant="proposer_critic_once")
    manager = RunManager(
        Pipeline(config=config), proposer_workers=args.workers, proposal_cache_dir=args.output_dir
    )
    manager._prepare_cache(documents)
    # executor.map keeps only a bounded window by processing slices explicitly.
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for start in range(0, len(documents), args.workers * 2):
            futures = [
                executor.submit(manager._propose, i, documents[i])
                for i in range(start, min(start + args.workers * 2, len(documents)))
            ]
            for future in futures:
                future.result()
    print(Path(args.output_dir).resolve())
    return 0
