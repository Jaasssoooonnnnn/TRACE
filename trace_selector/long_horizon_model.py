"""Long-horizon TRACE selector with shared document nodes."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import HeteroConv, SAGEConv

from .attention import AnswerAwareQFormer

EDGE_TYPES = (
    ("evidence", "belongs_to", "document"),
    ("document", "contains", "evidence"),
    ("subquery", "opens", "evidence"),
    ("evidence", "opened_by", "subquery"),
    ("subquery", "finds", "evidence"),
    ("evidence", "found_by", "subquery"),
    ("subquery", "next", "subquery"),
    ("subquery", "previous", "subquery"),
)
UPDATED_NODE_TYPES = ("subquery", "evidence", "document")


class LongHorizonSelector(nn.Module):
    def __init__(self, *, embedding_dim=4096):
        hidden_dim = 256
        num_layers = 4
        dropout = 0.1
        qformer_heads = 4
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.qformer_heads = qformer_heads

        def encoder():
            return nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )

        self.query_encoder = encoder()
        self.subquery_encoder = encoder()
        self.evidence_encoder = encoder()
        self.response_encoder = encoder()
        self.orphan_state = nn.Parameter(torch.zeros(1, hidden_dim))
        self.document_state = nn.Parameter(torch.zeros(1, hidden_dim))
        self.vote_projection = nn.Linear(2, hidden_dim, bias=False)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        edge_types = EDGE_TYPES
        self.updated_node_types = UPDATED_NODE_TYPES
        for _ in range(num_layers):
            self.convs.append(
                HeteroConv(
                    {
                        edge_type: SAGEConv((-1, -1), hidden_dim, root_weight=False)
                        for edge_type in edge_types
                    },
                    aggr="sum",
                )
            )
            self.norms.append(
                nn.ModuleDict(
                    {
                        node_type: nn.LayerNorm(hidden_dim)
                        for node_type in self.updated_node_types
                    }
                )
            )
        self.answer_qformer = AnswerAwareQFormer(
            hidden_dim, heads=qformer_heads, dropout=dropout
        )
        self.query_readout = nn.Linear(hidden_dim, hidden_dim)
        self.response_readout = nn.Linear(hidden_dim, hidden_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    def forward(self, data) -> torch.Tensor:
        query = self.query_encoder(data["query"].x.float())
        subquery = self.subquery_encoder(data["subquery"].x.float())
        subquery = (
            subquery + data["subquery"].is_orphan.unsqueeze(-1) * self.orphan_state
        )
        evidence = self.evidence_encoder(data["evidence"].x.float())
        response = self.response_encoder(data["response"].x.float())
        counts = data["response"].vote_count.float()
        response_batch = getattr(data["response"], "batch", None)
        if response_batch is None:
            trace_count = torch.full_like(counts, response.shape[0])
        else:
            graph_count = int(response_batch.max()) + 1
            totals = torch.zeros(graph_count, device=counts.device).scatter_add_(
                0, response_batch, torch.ones_like(counts)
            )
            trace_count = totals[response_batch]
        vote_features = torch.stack(
            (torch.log1p(counts), counts / trace_count.clamp_min(1.0)), dim=-1
        )
        response = response + self.vote_projection(vote_features)
        x_dict = {
            "query": query,
            "subquery": subquery,
            "evidence": evidence,
            "document": self.document_state.expand(data["document"].num_nodes, -1),
            "response": response,
        }
        for conv, norms in zip(self.convs, self.norms):
            updates = conv(x_dict, data.edge_index_dict)
            next_x = dict(x_dict)
            for node_type in self.updated_node_types:
                update = F.gelu(updates[node_type])
                update = F.dropout(update, p=self.dropout, training=self.training)
                next_x[node_type] = norms[node_type](x_dict[node_type] + update)
            x_dict = next_x
        x_dict["response"] = self.answer_qformer(
            x_dict["response"],
            x_dict["subquery"],
            x_dict["evidence"],
            data["subquery", "qformer_context", "response"].edge_index,
            data["evidence", "qformer_context", "response"].edge_index,
        )
        query = F.normalize(self.query_readout(x_dict["query"]), dim=-1)
        response = F.normalize(self.response_readout(x_dict["response"]), dim=-1)
        response_batch = getattr(data["response"], "batch", None)
        if response_batch is None:
            response_batch = torch.zeros(
                response.shape[0], dtype=torch.long, device=response.device
            )
        return self.logit_scale.exp().clamp(max=100.0) * (
            query[response_batch] * response
        ).sum(dim=-1)
