"""Browsing occurrence graphs with source-identity mean Doc nodes."""

import json
from pathlib import Path

import torch
from torch_geometric.data import HeteroData

from .data import apply_candidate_labels, edge_tensor
from .document import compile_question


class LongHorizonDataset(torch.utils.data.Dataset):
    def __init__(self, root, split="test"):
        root = Path(root)
        spec = json.loads((root / "manifest.json").read_text())[split]
        p = torch.load(root / spec["graphs"], map_location="cpu", weights_only=False)
        base = torch.load(
            root / spec["base_embeddings"],
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        sub = torch.load(
            root / spec["subquery_embeddings"],
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        mapping = json.loads((root / spec["document_identities"]).read_text())
        if any(
            (
                x["embedding_model"]
                not in ("Qwen/Qwen3-Embedding-8B", "synthetic-demo")
                or x["embedding_dim"] != 4096
                for x in (base, sub)
            )
        ):
            raise ValueError(
                "Expected Qwen3-Embedding-8B vectors with 4096 dimensions"
            )
        graphs = [g for g in p["graphs"] if g.get("split", "test") == split]
        if "cell" in spec:
            graphs = [g for g in graphs if g["cell"] == spec["cell"]]
        offset = int(spec.get("qid_offset", 0))
        if "question_ids" in spec:
            ids = set(json.loads((root / spec["question_ids"]).read_text()))
            graphs = [g for g in graphs if int(g["qid"]) - offset in ids]
            if {int(g["qid"]) - offset for g in graphs} != ids:
                raise ValueError("Question subset does not match this pool")
        self.total_questions = int(spec.get("total_questions", len(graphs)))
        if split == "test":
            apply_candidate_labels(graphs, root / spec["labels"])
        self.empty_questions = [int(g["qid"]) for g in graphs if not g["responses"]]
        self.graphs = [g for g in graphs if g["responses"]]
        if len(graphs) != self.total_questions:
            raise ValueError("Manifest denominator does not match graph coverage")
        self.base_embeddings = base["embeddings"]
        self.subquery_embeddings = sub["embeddings"]
        self.embedding_dim = 4096
        self.compiled = [
            compile_question(g, mapping["questions"][str(g["qid"])])
            for g in self.graphs
        ]
        self.node_type = "response"
        self.label_key = "y"

    def __len__(self):
        return len(self.graphs)

    def labels(self, index):
        return torch.tensor(
            [r["correct"] for r in self.graphs[index]["responses"]], dtype=torch.float32
        )

    def __getitem__(self, index: int) -> HeteroData:
        graph = self.graphs[index]
        data = HeteroData()
        data.qid = torch.tensor([graph["qid"]], dtype=torch.long)
        data["query"].x = self.base_embeddings[[graph["query_index"]]]
        responses = graph["responses"]
        data["response"].x = self.base_embeddings[
            torch.tensor([row["response_index"] for row in responses], dtype=torch.long)
        ]
        data["response"].y = torch.tensor(
            [row["correct"] for row in responses], dtype=torch.float32
        )
        data["response"].vote_count = torch.tensor(
            [row["vote_count"] for row in responses], dtype=torch.long
        )
        data["response"].seed = torch.tensor(
            [row["seed"] for row in responses], dtype=torch.long
        )
        selected_observations = [r["observations"] for r in responses]
        subqueries = []
        subquery_lookup = {}
        subquery_to_response = []
        temporal_next = []
        for response_index, response in enumerate(responses):
            previous = None
            for row in response["subqueries"]:
                local_index = len(subqueries)
                subqueries.append(row)
                subquery_lookup[response_index, row["subquery_id"]] = local_index
                subquery_to_response.append((local_index, response_index))
                if previous is not None:
                    temporal_next.append((previous, local_index))
                previous = local_index
        valid_query_rows = [
            row["query_index"] for row in subqueries if row["query_index"] >= 0
        ]
        subquery_x = torch.zeros(
            (len(subqueries), self.embedding_dim), dtype=self.subquery_embeddings.dtype
        )
        if valid_query_rows:
            positions = [
                i for i, row in enumerate(subqueries) if row["query_index"] >= 0
            ]
            subquery_x[torch.tensor(positions, dtype=torch.long)] = (
                self.subquery_embeddings[
                    torch.tensor(valid_query_rows, dtype=torch.long)
                ]
            )
        data["subquery"].x = subquery_x
        data["subquery"].is_orphan = torch.tensor(
            [row["is_orphan"] for row in subqueries], dtype=torch.float32
        )
        observations = []
        observation_sources = []
        observation_responses = []
        for response_index, observations_for_response in enumerate(
            selected_observations
        ):
            for observation in observations_for_response:
                observations.append(observation)
                observation_sources.append(
                    subquery_lookup[response_index, observation["subquery_id"]]
                )
                observation_responses.append(response_index)
        observation_evidence = list(range(len(observations)))
        if observations:
            data["evidence"].x = self.base_embeddings[
                torch.tensor(
                    [row["content_index"] for row in observations], dtype=torch.long
                )
            ]
        else:
            data["evidence"].x = torch.empty(
                (0, self.embedding_dim), dtype=self.base_embeddings.dtype
            )
        q = self.compiled[index]
        data["document"].num_nodes = q["capacity"]
        edge = torch.tensor([list(range(q["evidence"])), q["slots"]], dtype=torch.long)
        data["evidence", "belongs_to", "document"].edge_index = edge
        data["document", "contains", "evidence"].edge_index = edge.flip(0)
        opened = []
        found = []
        for subquery_index, evidence_id, observation in zip(
            observation_sources, observation_evidence, observations
        ):
            edges = opened if int(observation["tool_kind"]) == 0 else found
            edges.append((subquery_index, evidence_id))
        data["subquery", "opens", "evidence"].edge_index = edge_tensor(opened)
        data["evidence", "opened_by", "subquery"].edge_index = edge_tensor(
            [(target, source) for source, target in opened]
        )
        data["subquery", "finds", "evidence"].edge_index = edge_tensor(found)
        data["evidence", "found_by", "subquery"].edge_index = edge_tensor(
            [(target, source) for source, target in found]
        )
        data["subquery", "next", "subquery"].edge_index = edge_tensor(temporal_next)
        data["subquery", "previous", "subquery"].edge_index = edge_tensor(
            [(target, source) for source, target in temporal_next]
        )
        data["subquery", "qformer_context", "response"].edge_index = edge_tensor(
            subquery_to_response
        )
        data["evidence", "qformer_context", "response"].edge_index = edge_tensor(
            list(zip(observation_evidence, observation_responses))
        )
        data.graph_index = torch.tensor([index], dtype=torch.long)
        return data
