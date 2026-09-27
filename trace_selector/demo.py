"""Create small synthetic input caches for running the complete pipeline on CPU."""

import argparse
import json
from pathlib import Path

import torch

from .data import question_split

MODEL = "synthetic-demo"
DIM = 4096


def payload(graphs, **extra):
    return {"embedding_model": MODEL, "embedding_dim": DIM, "graphs": graphs, **extra}


def make_webqa(root):
    training = []
    counts = {"train": 0, "validation": 0}
    index = 0
    while min(counts.values()) < 2:
        question = f"Synthetic question {index}"
        split = question_split(question)
        if counts[split] < 2:
            training.append((question, split))
            counts[split] += 1
        index += 1
    for directory, questions in [
        ("train", training),
        ("test", [("Synthetic test 0", "test"), ("Synthetic test 1", "test")]),
    ]:
        graphs, metadata, sub_entries, chunk_entries = [], [], [], []
        sub_vectors, chunk_vectors = [], []
        for i, (question, split) in enumerate(questions):
            gid = f"demo:nq:{directory}:{i}"
            graphs.append(
                {
                    "graph_id": gid,
                    "question": question,
                    "split": split,
                    "trace_count": 3,
                    "sample_ids": [0, 1, 2],
                    "query_x": torch.randn(1, DIM),
                    "answer_x": torch.randn(2, DIM),
                    "answer_em": torch.tensor([1.0, 0.0]),
                    "answer_f1": torch.tensor([1.0, 0.0]),
                    "answer_vote_counts": torch.tensor([2, 1]),
                    "answer_texts": ["A", "B"],
                }
            )
            metadata.append(
                {
                    "graph_id": gid,
                    "trace_count": 3,
                    "event_sample_ids": [0, 1, 2],
                    "event_answer_ids": [0, 1, 0],
                    "event_subquery_text_ids": [0, 1, 2],
                    "event_search_indices": [0, 0, 0],
                    "subquery_occurrence_to_event": [[0, 0], [1, 1], [2, 2]],
                    "chunk_cross_rollout_coverage": [3, 3, 3],
                    "event_to_event": [],
                    "event_chunk_content_ids": [[0, 1, 2]] * 3,
                    "event_chunk_ranks": [[1, 2, 3]] * 3,
                }
            )
            entry = {
                "graph_id": gid,
                "embedding_indices": torch.arange(i * 3, i * 3 + 3),
            }
            sub_entries.append(entry)
            chunk_entries.append(entry)
            sub_vectors.append(torch.randn(3, DIM))
            chunk_vectors.append(torch.randn(3, DIM))
        dest = root / directory
        for d in [
            "tensor_graphs",
            "metadata",
            "subquery_embeddings",
            "chunk_embeddings",
        ]:
            (dest / d).mkdir(parents=True)
        if directory == "train":
            torch.save(payload(graphs), dest / "tensor_graphs/part00.pt")
        else:
            for name in [
                "nq",
                "hotpotqa",
                "triviaqa",
                "popqa",
                "2wiki",
                "musique",
                "bamboogle",
            ]:
                torch.save(
                    payload(graphs if name == "nq" else []),
                    dest / f"tensor_graphs/{name}.pt",
                )
        (dest / "metadata/part00.jsonl").write_text(
            "".join(json.dumps(x) + "\n" for x in metadata)
        )
        for kind, entries, vectors in [
            ("subquery", sub_entries, sub_vectors),
            ("chunk", chunk_entries, chunk_vectors),
        ]:
            torch.save(
                payload(
                    entries,
                    schema_version="retrieval_text_embeddings_v1",
                    node_kind=kind,
                    embeddings=torch.cat(vectors),
                ),
                dest / f"{kind}_embeddings/part00.pt",
            )
    (root / "manifest.json").write_text(json.dumps({"test": {"total_questions": 2}}))


def make_long_horizon(root):
    root.mkdir(parents=True)
    graphs, mapping = [], {}
    for qid, split in enumerate(
        ["train", "train", "validation", "validation", "test", "test"]
    ):
        responses = []
        for i in range(3):
            responses.append(
                {
                    "seed": 42 + i,
                    "sample_id": i,
                    "correct": i != 1,
                    "response_index": 1 + i,
                    "vote_count": 2 if i != 1 else 1,
                    "answer": "A" if i != 1 else "B",
                    "subqueries": [
                        {"subquery_id": 0, "query_index": i, "is_orphan": False}
                    ],
                    "observations": [
                        {
                            "event_id": 1,
                            "tool_kind": i % 2,
                            "content_index": 4 + i,
                            "doc_index": 0,
                            "subquery_id": 0,
                        }
                    ],
                }
            )
        graphs.append(
            {"qid": qid, "split": split, "query_index": 0, "responses": responses}
        )
        mapping[str(qid)] = {
            "evidence_keys": [[42 + i, 1] for i in range(3)],
            "old_doc_indices": [0, 0, 0],
            "original_doc_indices": [0],
            "identities": [{"key": "url:https://example.org/demo", "verified": True}],
            "evidence_identity_indices": [0, 0, 0],
            "verified_mask": [True, True, True],
        }
    torch.save({"graphs": graphs}, root / "graphs.pt")
    torch.save(payload([], embeddings=torch.randn(7, DIM)), root / "base_embeddings.pt")
    torch.save(
        payload([], embeddings=torch.randn(3, DIM)), root / "subquery_embeddings.pt"
    )
    (root / "document_identities.json").write_text(json.dumps({"questions": mapping}))
    fields = {
        "graphs": "graphs.pt",
        "base_embeddings": "base_embeddings.pt",
        "subquery_embeddings": "subquery_embeddings.pt",
        "document_identities": "document_identities.json",
    }
    manifest = {split: dict(fields) for split in ["train", "validation", "test"]}
    manifest["test"]["labels"] = "labels.jsonl"
    (root / "labels.jsonl").write_text(
        "".join(
            json.dumps({"qid": qid, "sample_id": i, "correct": i != 1}) + "\n"
            for qid in [4, 5]
            for i in range(3)
        )
    )
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.manual_seed(42)
    make_webqa(args.output / "webqa")
    make_long_horizon(args.output / "long_horizon")
    print(args.output)


if __name__ == "__main__":
    main()
