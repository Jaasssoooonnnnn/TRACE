"""Convert raw completed rollouts into text pools with retrieval provenance."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from .trajectories import build_browser_pool, build_webqa_pool


def read_records(paths):
    for path in paths:
        path = Path(path)
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq

            for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
                yield from batch.to_pylist()
        else:
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        yield json.loads(line)


def prepare_pools(records, task, dataset, split, k, corpus=None):
    if k <= 0:
        raise ValueError("Rollout count must be positive")
    groups = defaultdict(list)
    for row in records:
        key = int(row["qid"] if task == "long_horizon" else row["question_index"])
        groups[key].append(row)
    builder = build_webqa_pool if task == "webqa" else build_browser_pool
    for key in sorted(groups):
        kwargs = {"corpus": corpus} if task == "long_horizon" else {}
        pool = builder(groups[key], dataset, split, k, **kwargs)
        if pool is not None:
            yield pool


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("webqa", "long_horizon"))
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--corpus", type=Path, nargs="*", default=[])
    parser.add_argument(
        "--labels", type=Path, help="Saved JSONL qid/sample_id/correct judgments"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    corpus = {}
    for record in read_records(args.corpus):
        key = record.get("text_hash", record.get("hash"))
        if key is None or (key in corpus and corpus[key] != record["text"]):
            raise ValueError("Conflicting or missing corpus text hash")
        if hashlib.sha256(record["text"].encode("utf-8")).hexdigest() != key:
            raise ValueError("Corpus text does not match its declared hash")
        corpus[key] = record["text"]
    records = list(read_records(args.input))
    if args.labels:
        if args.task != "long_horizon":
            raise ValueError("WebQA labels are computed from golden_answers")
        judgments = {}
        for label in read_records([args.labels]):
            key = (int(label["qid"]), int(label["sample_id"]))
            if not isinstance(label["correct"], bool) or key in judgments:
                raise ValueError("Judgments must be unique boolean labels")
            judgments[key] = label["correct"]
        for record in records:
            sample = int(
                record.get(
                    "sample_id",
                    record.get("traj_idx", int(record.get("seed", 42)) - 42),
                )
            )
            record["correct"] = judgments[int(record["qid"]), sample]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for pool in prepare_pools(
            records, args.task, args.dataset, args.split, args.k, corpus
        ):
            stream.write(json.dumps(pool, ensure_ascii=False) + "\n")
            count += 1
    temporary.replace(args.output)
    print(json.dumps({"questions": count, "task": args.task}))


if __name__ == "__main__":
    main()
