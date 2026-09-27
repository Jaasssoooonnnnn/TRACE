"""Frozen embedding sidecars and deterministic question-level splitting."""

import hashlib
import json
import unicodedata
from pathlib import Path

import torch

EMBEDDING_SCHEMA_VERSION = "retrieval_text_embeddings_v1"


def edge_tensor(edges):
    return (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges
        else torch.empty((2, 0), dtype=torch.long)
    )


def question_split(question):
    normalized = " ".join(unicodedata.normalize("NFC", str(question or "")).split())
    digest = hashlib.sha256(
        f"gnn-selector-question-split|42|{normalized}".encode()
    ).digest()
    return "train" if int.from_bytes(digest[:8], "big") / 2**64 < 0.95 else "validation"


def read_metadata(paths):
    result = {}
    for path in paths:
        with path.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["graph_id"] in result:
                    raise ValueError(f"Duplicate graph: {row['graph_id']}")
                result[row["graph_id"]] = row
    return result


class EmbeddingStore:
    def __init__(self, paths: list[Path], expected_kind: str):
        self.expected_kind = expected_kind
        self.embedding_model = None
        self.embedding_dim = None
        self._by_graph_id = {}
        self._payloads = []
        for path in paths:
            payload = torch.load(
                path, map_location="cpu", weights_only=False, mmap=True
            )
            if payload.get("schema_version") != EMBEDDING_SCHEMA_VERSION:
                raise ValueError(f"unsupported embedding sidecar: {path}")
            if payload.get("node_kind") != expected_kind:
                raise ValueError(
                    f"expected {expected_kind} sidecar, found {payload.get('node_kind')}: {path}"
                )
            model = payload["embedding_model"]
            dimension = int(payload["embedding_dim"])
            if self.embedding_model is None:
                self.embedding_model = model
                self.embedding_dim = dimension
            elif (model, dimension) != (self.embedding_model, self.embedding_dim):
                raise ValueError(f"embedding configuration differs in {path}")
            payload_index = len(self._payloads)
            self._payloads.append(payload)
            for graph in payload["graphs"]:
                graph_id = graph["graph_id"]
                if graph_id in self._by_graph_id:
                    raise ValueError(f"duplicate graph ID: {graph_id}")
                self._by_graph_id[graph_id] = (
                    payload_index,
                    graph["embedding_indices"],
                )

    def __contains__(self, graph_id: str) -> bool:
        return graph_id in self._by_graph_id

    def features(self, graph_id: str) -> torch.Tensor:
        payload_index, indices = self._by_graph_id[graph_id]
        table = self._payloads[payload_index]["embeddings"]
        return table[indices.long()]

    def __len__(self) -> int:
        return len(self._by_graph_id)


def apply_candidate_labels(graphs, path):
    if path.suffix == ".json":
        by_question = json.loads(path.read_text())
        if not by_question:
            raise ValueError("Empty candidate judgment file")
        prefix = next(iter(by_question)).rsplit("/", 1)[0]
        for graph in graphs:
            key = graph.get("key") or f"{prefix}/{int(graph['qid']):05d}"
            records = by_question[key]
            by_sample = {int(row["sample_id"]): row for row in records}
            if len(by_sample) != len(records):
                raise ValueError(f"Duplicate candidate in {key}")
            for response in graph["responses"]:
                sample = int(response.get("sample_id", int(response["seed"]) - 42))
                row = by_sample[sample]
                if not isinstance(row["correct"], bool):
                    raise ValueError("Candidate correctness must be boolean")
                if (
                    "candidate_hash" in row
                    and response["candidate_hash"] != row["candidate_hash"]
                ):
                    raise ValueError(f"Candidate hash mismatch: {key}/{sample}")
                response["correct"] = row["correct"]
    else:
        labels = {}
        with path.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (int(row["qid"]), int(row["sample_id"]))
                if not isinstance(row["correct"], bool):
                    raise ValueError("Candidate correctness must be boolean")
                if key in labels and labels[key] != row["correct"]:
                    raise ValueError(f"Conflicting candidate judgment: {key}")
                labels[key] = row["correct"]
        for graph in graphs:
            for response in graph["responses"]:
                sample = int(response.get("sample_id", int(response["seed"]) - 42))
                response["correct"] = labels[(int(graph["qid"]), sample)]
