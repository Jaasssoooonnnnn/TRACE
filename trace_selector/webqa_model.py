"""WebQA TRACE selector."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.utils import scatter

from .attention import AnswerAwareQFormer

EDGE_TYPES = (
    ("query", "starts", "subquery"),
    ("subquery", "next", "subquery"),
    *(("subquery", f"rank{r}", "chunk_occurrence") for r in (1, 2, 3)),
    *(("chunk_occurrence", f"rank{r}_of", "subquery") for r in (1, 2, 3)),
    ("chunk_occurrence", "same_content_intra_trace", "chunk_occurrence"),
    ("chunk_occurrence", "same_content_cross_trace", "chunk_occurrence"),
)


class WebQASelector(nn.Module):
    def __init__(self, embedding_dim=4096):
        super().__init__()
        self.embedding_dim = embedding_dim

        def encoder():
            return nn.Sequential(
                nn.Linear(embedding_dim, 256), nn.LayerNorm(256), nn.GELU()
            )

        self.query_encoder = encoder()
        self.answer_encoder = encoder()
        # Preserve the parameter initialization sequence used for WebQA training.
        self.answer_encoder[0].reset_parameters()
        self.subquery_encoder = encoder()
        self.chunk_encoder = encoder()
        self.answer_qformer = AnswerAwareQFormer(256, heads=4, dropout=0.1)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(4):
            self.convs.append(
                HeteroConv(
                    {e: SAGEConv((-1, -1), 256, root_weight=False) for e in EDGE_TYPES},
                    aggr="sum",
                )
            )
            self.norms.append(
                nn.ModuleDict(
                    {n: nn.LayerNorm(256) for n in ("subquery", "chunk_occurrence")}
                )
            )
        self.query_readout = nn.Linear(256, 256)
        self.answer_readout = nn.Linear(256, 256)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.answer_occurrence_weight = nn.Parameter(torch.zeros(256, 2))

    def forward(self, data):
        x = {
            "query": self.query_encoder(data["query"].x.float()),
            "answer": self.answer_encoder(data["answer"].x.float()),
            "subquery": self.subquery_encoder(data["subquery"].x.float()),
        }
        chunks = self.chunk_encoder(data["chunk"].x.float())
        content, occurrence = data["chunk", "indexes", "chunk_occurrence"].edge_index
        count = data["chunk_occurrence"].num_nodes
        if content.numel() != count:
            raise ValueError(
                "Each evidence occurrence needs exactly one content embedding"
            )
        x["chunk_occurrence"] = scatter(
            chunks[content], occurrence, dim=0, dim_size=count, reduce="sum"
        )
        x["answer"] = x["answer"] + F.linear(
            data["answer"].occurrence.float(), self.answer_occurrence_weight
        )
        for conv, norms in zip(self.convs, self.norms):
            updates = conv(x, data.edge_index_dict)
            x = {
                **x,
                **{
                    n: norm(
                        x[n]
                        + F.dropout(F.gelu(updates[n]), p=0.1, training=self.training)
                    )
                    for n, norm in norms.items()
                },
            }
        answer = self.answer_qformer(
            x["answer"],
            x["subquery"],
            x["chunk_occurrence"],
            data["subquery", "qformer_context", "answer"].edge_index,
            data["chunk_occurrence", "qformer_context", "answer"].edge_index,
        )
        batch = getattr(data["answer"], "batch", None)
        if batch is None:
            batch = torch.zeros(answer.shape[0], dtype=torch.long, device=answer.device)
        query = F.normalize(self.query_readout(x["query"])[batch], p=2, dim=-1)
        answer = F.normalize(self.answer_readout(answer), p=2, dim=-1)
        return self.logit_scale.clamp(max=math.log(100.0)).exp() * (query * answer).sum(
            -1
        )
