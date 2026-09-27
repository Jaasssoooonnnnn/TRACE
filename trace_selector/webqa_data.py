"""Occurrence-preserving WebQA graphs from frozen tensor/metadata caches."""

import json
from pathlib import Path

import torch
from torch_geometric.data import HeteroData

from .data import EmbeddingStore, edge_tensor, question_split, read_metadata


def resolved_event_answer_ids(metadata: dict, graph_id: str) -> list[int]:
    event_count = len(metadata["event_sample_ids"])
    answer_ids = metadata.get("event_answer_ids")
    if answer_ids is not None:
        answer_ids = list(map(int, answer_ids))
    else:
        answer_ids = [-1] * event_count
        for event_id, answer_id in metadata["event_to_answer"]:
            event_id = int(event_id)
            if answer_ids[event_id] >= 0:
                raise ValueError(f"multiple answers for {graph_id} event={event_id}")
            answer_ids[event_id] = int(answer_id)
    if len(answer_ids) != event_count or any(
        (answer_id < 0 for answer_id in answer_ids)
    ):
        raise ValueError(f"missing event-to-answer mapping for {graph_id}")
    return answer_ids


class WebQADataset(torch.utils.data.Dataset):
    def __init__(self, root, split="test"):
        manifest_path = Path(root) / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        )
        root = Path(root) / ("test" if split == "test" else "train")
        names = ("nq", "hotpotqa", "triviaqa", "popqa", "2wiki", "musique", "bamboogle")
        paths = (
            [root / "tensor_graphs" / f"{n}.pt" for n in names]
            if split == "test"
            else sorted((root / "tensor_graphs").glob("part*.pt"))
        )
        if not paths or any((not p.is_file() for p in paths)):
            raise FileNotFoundError(f"Missing tensor graphs under {root}")
        self.retrieval_metadata = read_metadata(
            sorted((root / "metadata").glob("*.jsonl"))
        )
        self.subquery_embeddings = EmbeddingStore(
            sorted((root / "subquery_embeddings").glob("*.pt")), "subquery"
        )
        self.chunk_embeddings = EmbeddingStore(
            sorted((root / "chunk_embeddings").glob("*.pt")), "chunk"
        )
        self.graphs = []
        for path in paths:
            p = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            if (
                p["embedding_model"]
                not in ("Qwen/Qwen3-Embedding-8B", "synthetic-demo")
                or p["embedding_dim"] != 4096
            ):
                raise ValueError(
                    "Expected Qwen3-Embedding-8B vectors with 4096 dimensions"
                )
            self.embedding_dim = int(p["embedding_dim"])
            for store in (self.subquery_embeddings, self.chunk_embeddings):
                if (store.embedding_model, store.embedding_dim) != (
                    p["embedding_model"],
                    p["embedding_dim"],
                ):
                    raise ValueError(
                        "Graph and sidecar embedding configurations differ"
                    )
            for g in p["graphs"]:
                keep = (
                    g["split"] == "test"
                    if split == "test"
                    else split is None or question_split(g["question"]) == split
                )
                if keep:
                    self.graphs.append(g)
        self.total_questions = (
            int(manifest.get("test", {}).get("total_questions", 3125))
            if split == "test"
            else len(self.graphs)
        )
        self._heterodata_cache = {}
        for graph in self.graphs:
            gid = graph["graph_id"]
            if any(
                (
                    gid not in store
                    for store in (
                        self.retrieval_metadata,
                        self.subquery_embeddings,
                        self.chunk_embeddings,
                    )
                )
            ):
                raise ValueError(f"Missing sidecar for {gid}")
        self.node_type = "answer"
        self.label_key = "y_em"

    def subset(self, split):
        dataset = object.__new__(WebQADataset)
        dataset.__dict__.update(self.__dict__)
        dataset.graphs = [
            g for g in self.graphs if question_split(g["question"]) == split
        ]
        dataset.total_questions = len(dataset.graphs)
        return dataset

    def __len__(self):
        return len(self.graphs)

    def labels(self, index):
        graph = self.graphs[index]
        return graph["answer_em"][self.answer_canonical_ids(graph)]

    def answer_canonical_ids(
        self, graph: dict, metadata: dict | None = None
    ) -> list[int]:
        if metadata is None:
            metadata = self.retrieval_metadata[graph["graph_id"]]
        sample_to_answer = {}
        for sample_id, answer_id in zip(
            map(int, metadata["event_sample_ids"]),
            resolved_event_answer_ids(metadata, graph["graph_id"]),
        ):
            previous = sample_to_answer.setdefault(sample_id, answer_id)
            if previous != answer_id:
                raise ValueError(
                    f"multiple candidate answers for {graph['graph_id']} sample_id={sample_id}"
                )
        graph_sample_ids = list(map(int, graph["sample_ids"]))
        missing = [
            sample_id
            for sample_id in graph_sample_ids
            if sample_id not in sample_to_answer
        ]
        if missing:
            raise ValueError(
                f"missing candidate answers for {graph['graph_id']}: {missing[:3]}"
            )
        return [sample_to_answer[sample_id] for sample_id in graph_sample_ids]

    def __getitem__(self, index: int) -> HeteroData:
        graph = self.graphs[index]
        cached = self._heterodata_cache.get(graph["graph_id"])
        if cached is not None:
            return cached
        data = HeteroData()
        data["query"].x = graph["query_x"]
        metadata = self.retrieval_metadata[graph["graph_id"]]
        if int(metadata["trace_count"]) != int(graph["trace_count"]):
            raise ValueError(f"trace count differs for {graph['graph_id']}")
        canonical_ids = self.answer_canonical_ids(graph, metadata)
        canonical_id_tensor = torch.tensor(canonical_ids, dtype=torch.long)
        data["answer"].x = graph["answer_x"][canonical_id_tensor]
        data["answer"].y_em = graph["answer_em"][canonical_id_tensor]
        data["answer"].y_f1 = graph["answer_f1"][canonical_id_tensor]
        data["answer"].vote_count = graph["answer_vote_counts"][canonical_id_tensor]
        data["answer"].canonical_id = canonical_id_tensor
        data["answer"].sample_id = torch.tensor(
            list(map(int, graph["sample_ids"])), dtype=torch.long
        )
        trace_count = max(int(graph["trace_count"]), 1)
        answer_counts = data["answer"].vote_count.float()
        data["answer"].occurrence = torch.stack(
            (torch.log1p(answer_counts), answer_counts / trace_count), dim=-1
        )
        assert metadata is not None
        subquery_table = self.subquery_embeddings.features(graph["graph_id"])
        event_subquery_ids = torch.tensor(
            metadata["event_subquery_text_ids"], dtype=torch.long
        )
        valid_subquery_ids = event_subquery_ids[event_subquery_ids >= 0]
        data["subquery"].x = subquery_table[valid_subquery_ids]
        chunk_table = self.chunk_embeddings.features(graph["graph_id"])
        data["chunk"].x = chunk_table
        event_count = len(metadata["event_sample_ids"])
        chunk_counts = torch.tensor(
            metadata["chunk_cross_rollout_coverage"], dtype=torch.float32
        )
        data["chunk"].occurrence = torch.stack(
            (torch.log1p(chunk_counts), chunk_counts / trace_count), dim=-1
        )
        subquery_occurrence_x = data["subquery"].x
        event_subquery_x = torch.zeros(
            (event_count, subquery_occurrence_x.shape[1]),
            dtype=subquery_occurrence_x.dtype,
        )
        seen_subquery_events = set()
        for subquery_id, event_id in metadata["subquery_occurrence_to_event"]:
            event_id = int(event_id)
            if event_id in seen_subquery_events:
                raise ValueError(
                    f"multiple Subqueries for {graph['graph_id']} event={event_id}"
                )
            seen_subquery_events.add(event_id)
            event_subquery_x[event_id] = subquery_occurrence_x[subquery_id]
        data["subquery"].x = event_subquery_x
        data["query", "starts", "subquery"].edge_index = edge_tensor(
            [
                [0, event_id]
                for event_id, search_index in enumerate(
                    metadata["event_search_indices"]
                )
                if int(search_index) == 0
            ]
        )
        data["subquery", "next", "subquery"].edge_index = edge_tensor(
            metadata["event_to_event"]
        )
        answer_node_by_sample = {
            sample_id: answer_id
            for answer_id, sample_id in enumerate(map(int, graph["sample_ids"]))
        }
        target_answer_ids = [
            answer_node_by_sample[int(sample_id)]
            for sample_id in metadata["event_sample_ids"]
        ]
        subquery_to_answer = [
            [event_id, answer_id]
            for event_id, answer_id in enumerate(target_answer_ids)
        ]
        occurrence_content_ids = []
        occurrence_trace_ids = []
        occurrence_ranks = []
        occurrence_to_answer = []
        occurrences_by_content = {}
        rank_edges = {rank: [] for rank in (1, 2, 3)}
        sample_ids = list(map(int, metadata["event_sample_ids"]))
        event_answer_ids = [-1] * event_count
        for event_id, answer_id in subquery_to_answer:
            if event_answer_ids[event_id] >= 0:
                raise ValueError(
                    f"multiple answers for {graph['graph_id']} event={event_id}"
                )
            event_answer_ids[event_id] = answer_id
        if any((answer_id < 0 for answer_id in event_answer_ids)):
            raise ValueError(f"missing event-to-answer mapping for {graph['graph_id']}")
        for event_id, (content_ids, ranks) in enumerate(
            zip(metadata["event_chunk_content_ids"], metadata["event_chunk_ranks"])
        ):
            if len(content_ids) != 3 or sorted(map(int, ranks)) != [1, 2, 3]:
                raise ValueError(
                    f"subquery-chunk-occurrence requires exactly ranks 1,2,3 for {graph['graph_id']} event={event_id}"
                )
            for content_id, rank in zip(content_ids, ranks):
                content_id = int(content_id)
                rank = int(rank)
                occurrence_id = len(occurrence_content_ids)
                occurrence_content_ids.append(content_id)
                occurrence_trace_ids.append(sample_ids[event_id])
                occurrence_ranks.append(rank)
                occurrence_to_answer.append([occurrence_id, event_answer_ids[event_id]])
                occurrences_by_content.setdefault(content_id, []).append(occurrence_id)
                rank_edges[rank].append([event_id, occurrence_id])
        occurrence_count = len(occurrence_content_ids)
        data["chunk_occurrence"].x = torch.ones(
            (occurrence_count, 1), dtype=torch.float32
        )
        data["chunk_occurrence"].content_id = torch.tensor(
            occurrence_content_ids, dtype=torch.long
        )
        data["chunk_occurrence"].rank = torch.tensor(occurrence_ranks, dtype=torch.long)
        data["chunk_occurrence"].trace_id = torch.tensor(
            occurrence_trace_ids, dtype=torch.long
        )
        data["chunk", "indexes", "chunk_occurrence"].edge_index = edge_tensor(
            [
                [content_id, occurrence_id]
                for occurrence_id, content_id in enumerate(occurrence_content_ids)
            ]
        )
        for rank in (1, 2, 3):
            edges = rank_edges[rank]
            data[
                "subquery", f"rank{rank}", "chunk_occurrence"
            ].edge_index = edge_tensor(edges)
            data[
                "chunk_occurrence", f"rank{rank}_of", "subquery"
            ].edge_index = edge_tensor(
                [[occurrence_id, event_id] for event_id, occurrence_id in edges]
            )
        intra_trace_edges = []
        cross_trace_edges = []
        for occurrence_ids in occurrences_by_content.values():
            for left_position, left_id in enumerate(occurrence_ids):
                for right_id in occurrence_ids[left_position + 1 :]:
                    target = (
                        intra_trace_edges
                        if occurrence_trace_ids[left_id]
                        == occurrence_trace_ids[right_id]
                        else cross_trace_edges
                    )
                    target.append([left_id, right_id])
                    target.append([right_id, left_id])
        data[
            "chunk_occurrence", "same_content_intra_trace", "chunk_occurrence"
        ].edge_index = edge_tensor(intra_trace_edges)
        data[
            "chunk_occurrence", "same_content_cross_trace", "chunk_occurrence"
        ].edge_index = edge_tensor(cross_trace_edges)
        data["subquery", "qformer_context", "answer"].edge_index = edge_tensor(
            subquery_to_answer
        )
        data["chunk_occurrence", "qformer_context", "answer"].edge_index = edge_tensor(
            occurrence_to_answer
        )
        data.graph_index = torch.tensor([index], dtype=torch.long)
        self._heterodata_cache[graph["graph_id"]] = data
        return data
