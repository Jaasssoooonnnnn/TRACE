"""Frozen Qwen3 text encoding and caches consumed by the TRACE loaders."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F

from .data import EMBEDDING_SCHEMA_VERSION
from .prepare import read_records
from .source_identity import build_document_identities

MODEL_ID = "Qwen/Qwen3-Embedding-8B"
DIMENSION = 4096
WEBQA_DATASETS = (
    "nq",
    "hotpotqa",
    "triviaqa",
    "popqa",
    "2wiki",
    "musique",
    "bamboogle",
)


def last_token_pool(hidden_states, attention_mask):
    """Select the final unmasked token with either left or right padding."""
    if not bool(attention_mask.any(dim=1).all()):
        raise ValueError("An encoded text has no non-padding tokens")
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    last = (
        positions.expand_as(attention_mask)
        .masked_fill(attention_mask == 0, -1)
        .max(dim=1)
        .values
    )
    return hidden_states[
        torch.arange(hidden_states.shape[0], device=hidden_states.device), last
    ]


def encode_texts(texts, tokenizer, model, device, batch_size=16, max_length=4096):
    if batch_size <= 0 or max_length <= 0:
        raise ValueError("Batch size and token limit must be positive")
    rows = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            tokens = tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            tokens = {key: value.to(device) for key, value in tokens.items()}
            output = model(**tokens)
            pooled = last_token_pool(output.last_hidden_state, tokens["attention_mask"])
            if pooled.shape[1] < DIMENSION:
                raise ValueError(
                    "Qwen3-Embedding-8B must supply 4096-dimensional states"
                )
            vectors = F.normalize(pooled[:, :DIMENSION].float(), p=2, dim=1)
            if not bool(torch.isfinite(vectors).all()):
                raise ValueError("Nonfinite text embedding")
            rows.append(vectors.cpu())
    return torch.cat(rows) if rows else torch.empty((0, DIMENSION), dtype=torch.float32)


class QwenEncoder:
    model_id = MODEL_ID

    def __init__(
        self,
        model_path=MODEL_ID,
        device="cuda",
        dtype="bfloat16",
        batch_size=16,
        max_length=4096,
        save_float32=False,
    ):
        from transformers import AutoModel, AutoTokenizer

        self.device = torch.device(device)
        self.batch_size = batch_size
        self.max_length = max_length
        self.storage_dtype = torch.float32 if save_float32 else torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        self.tokenizer.truncation_side = "right"
        self.model = AutoModel.from_pretrained(
            model_path, torch_dtype=getattr(torch, dtype), attn_implementation="sdpa"
        ).to(self.device)
        self.model.requires_grad_(False)
        self.model.eval()

    def __call__(self, texts):
        return encode_texts(
            texts,
            self.tokenizer,
            self.model,
            self.device,
            self.batch_size,
            self.max_length,
        ).to(self.storage_dtype)

    def metadata(self):
        return {
            "embedding_model": self.model_id,
            "embedding_dim": DIMENSION,
            "max_length": self.max_length,
            "padding_side": "left",
            "truncation_side": "right",
            "pooling": "last_non_padding_token",
            "normalization": "l2_float32",
            "instruction_prefix": None,
            "storage_dtype": str(self.storage_dtype).removeprefix("torch."),
        }


def _table(texts, encoder):
    unique = list(dict.fromkeys(texts))
    index = {text: i for i, text in enumerate(unique)}
    return encoder(unique), index


def _write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_webqa(pools, output, encoder, shard_size=512):
    if shard_size <= 0:
        raise ValueError("Shard size must be positive")
    groups = defaultdict(list)
    for pool in pools:
        if pool["dataset"] not in WEBQA_DATASETS:
            raise ValueError("Unknown WebQA dataset name")
        folder = "test" if pool["split"] == "test" else "train"
        groups[folder, pool["dataset"]].append(pool)
    for (folder, dataset), source in groups.items():
        root = output / folder
        for name in (
            "tensor_graphs",
            "metadata",
            "subquery_embeddings",
            "chunk_embeddings",
        ):
            (root / name).mkdir(parents=True, exist_ok=True)
        retained = [pool for pool in source if pool["sample_ids"]]
        size = max(len(retained), 1) if folder == "test" else shard_size
        for start in range(0, max(len(retained), 1), size):
            shard = retained[start : start + size]
            name = (
                dataset if folder == "test" else f"part-{dataset}-{start // size:05d}"
            )
            vectors, index = _table(
                [
                    text
                    for pool in shard
                    for text in [
                        pool["question"],
                        *pool["answer_texts"],
                        *pool["subquery_texts"],
                        *pool["chunk_texts"],
                    ]
                ],
                encoder,
            )
            graphs, metadata = [], []
            entries = {kind: [] for kind in ("subquery", "chunk")}
            for pool in shard:
                graph = {
                    key: pool[key]
                    for key in (
                        "graph_id",
                        "question",
                        "split",
                        "trace_count",
                        "sample_ids",
                        "answer_texts",
                    )
                }
                graph["query_x"] = vectors[[index[pool["question"]]]]
                graph["answer_x"] = vectors[[index[t] for t in pool["answer_texts"]]]
                for field in ("answer_em", "answer_f1", "answer_vote_counts"):
                    graph[field] = torch.tensor(
                        pool[field],
                        dtype=torch.long if field.endswith("counts") else torch.float32,
                    )
                graphs.append(graph)
                metadata.append(pool["metadata"])
            torch.save(
                {**encoder.metadata(), "graphs": graphs},
                root / f"tensor_graphs/{name}.pt",
            )
            (root / f"metadata/{name}.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in metadata),
                encoding="utf-8",
            )
            for kind in entries:
                texts = list(
                    dict.fromkeys(
                        text for pool in shard for text in pool[f"{kind}_texts"]
                    )
                )
                side_index = {text: i for i, text in enumerate(texts)}
                entries[kind] = [
                    {
                        "graph_id": pool["graph_id"],
                        "embedding_indices": torch.tensor(
                            [side_index[t] for t in pool[f"{kind}_texts"]],
                            dtype=torch.long,
                        ),
                    }
                    for pool in shard
                ]
                side_vectors = vectors[
                    torch.tensor([index[t] for t in texts], dtype=torch.long)
                ]
                torch.save(
                    {
                        **encoder.metadata(),
                        "schema_version": EMBEDDING_SCHEMA_VERSION,
                        "node_kind": kind,
                        "embeddings": side_vectors,
                        "graphs": entries[kind],
                    },
                    root / f"{kind}_embeddings/{name}.pt",
                )
    test_count = sum(pool["split"] == "test" for pool in pools)
    if test_count:
        for dataset in WEBQA_DATASETS:
            path = output / f"test/tensor_graphs/{dataset}.pt"
            if not path.exists():
                torch.save({**encoder.metadata(), "graphs": []}, path)
        _write_json(output / "manifest.json", {"test": {"total_questions": test_count}})


def write_long_horizon(pools, output, encoder):
    texts = [
        text
        for pool in pools
        for text in [
            pool["question"],
            *[r["answer"] for r in pool["responses"]],
            *[o["text"] for r in pool["responses"] for o in r["observations"]],
        ]
    ]
    base, index = _table(texts, encoder)
    subtexts = [
        s["text"]
        for pool in pools
        for r in pool["responses"]
        for s in r["subqueries"]
        if s["text"]
    ]
    sub, subindex = _table(subtexts, encoder)
    graphs = []
    for pool in pools:
        graph = {key: pool[key] for key in ("qid", "split", "question")}
        graph["query_index"] = index[pool["question"]]
        graph["responses"] = []
        for response in pool["responses"]:
            row = {
                key: response[key]
                for key in ("seed", "sample_id", "correct", "vote_count", "answer")
            }
            row["response_index"] = index[response["answer"]]
            row["subqueries"] = [
                {
                    "subquery_id": s["subquery_id"],
                    "is_orphan": s["is_orphan"],
                    "query_index": subindex[s["text"]] if s["text"] else -1,
                }
                for s in response["subqueries"]
            ]
            row["observations"] = [
                {
                    **{
                        key: o[key]
                        for key in ("event_id", "tool_kind", "doc_index", "subquery_id")
                    },
                    "content_index": index[o["text"]],
                }
                for o in response["observations"]
            ]
            graph["responses"].append(row)
        graphs.append(graph)
    torch.save({"graphs": graphs}, output / "graphs.pt")
    torch.save(
        {**encoder.metadata(), "embeddings": base}, output / "base_embeddings.pt"
    )
    torch.save(
        {**encoder.metadata(), "embeddings": sub}, output / "subquery_embeddings.pt"
    )
    _write_json(output / "document_identities.json", build_document_identities(pools))
    judgments = [
        {"qid": p["qid"], "sample_id": r["sample_id"], "correct": r["correct"]}
        for p in pools
        if p["split"] == "test"
        for r in p["responses"]
    ]
    (output / "judgments.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in judgments), encoding="utf-8"
    )
    manifest = {}
    for split in sorted({p["split"] for p in pools}):
        manifest[split] = {
            "graphs": "graphs.pt",
            "base_embeddings": "base_embeddings.pt",
            "subquery_embeddings": "subquery_embeddings.pt",
            "document_identities": "document_identities.json",
        }
        if split == "test":
            manifest[split]["labels"] = "judgments.jsonl"
    _write_json(output / "manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("webqa", "long_horizon"))
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", default=MODEL_ID)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--save-float32", action="store_true")
    parser.add_argument("--shard-size", type=int, default=512)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use an empty cache output directory")
    pools = list(read_records(args.input))
    if not pools or any(pool["task"] != args.task for pool in pools):
        raise ValueError("Supply nonempty parsed pools of the selected task")
    ids = [p["graph_id"] if args.task == "webqa" else p["qid"] for p in pools]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate question IDs in parsed pools")
    encoder = QwenEncoder(
        args.model_path,
        args.device,
        args.dtype,
        args.batch_size,
        args.max_length,
        args.save_float32,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    if args.task == "webqa":
        write_webqa(pools, args.output, encoder, args.shard_size)
    else:
        write_long_horizon(pools, args.output, encoder)
    print(json.dumps({"questions": len(pools), "task": args.task}))


if __name__ == "__main__":
    main()
