"""Answer-conditioned, trajectory-local cross-attention."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import scatter, softmax


class AnswerAwareQFormer(nn.Module):
    """Candidate-conditioned cross-attention over Subquery and Evidence states."""

    def __init__(self, hidden_dim: int, *, heads: int, dropout: float):
        super().__init__()
        if heads <= 0 or hidden_dim % heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by qformer heads={heads}"
            )
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.query_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.context_type = nn.Parameter(torch.zeros(2, hidden_dim))
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.dropout = dropout

    def forward(
        self,
        answer: torch.Tensor,
        subquery: torch.Tensor,
        chunk_occurrence: torch.Tensor,
        subquery_context: torch.Tensor,
        chunk_context: torch.Tensor,
    ) -> torch.Tensor:
        context = torch.cat(
            (subquery + self.context_type[0], chunk_occurrence + self.context_type[1]),
            dim=0,
        )
        source = torch.cat((subquery_context[0], chunk_context[0] + subquery.shape[0]))
        target = torch.cat((subquery_context[1], chunk_context[1]))
        query = self.query_projection(answer).reshape(
            answer.shape[0], self.heads, self.head_dim
        )
        key = self.key_projection(context).reshape(
            context.shape[0], self.heads, self.head_dim
        )
        value = self.value_projection(context).reshape(
            context.shape[0], self.heads, self.head_dim
        )
        logits = (query[target] * key[source]).sum(dim=-1) / math.sqrt(self.head_dim)
        attention = softmax(logits, target, num_nodes=answer.shape[0])
        messages = attention.unsqueeze(-1) * value[source]
        pooled = scatter(
            messages, target, dim=0, dim_size=answer.shape[0], reduce="sum"
        ).reshape(answer.shape[0], -1)
        answer = self.attention_norm(
            answer
            + F.dropout(
                self.output_projection(pooled), p=self.dropout, training=self.training
            )
        )
        return self.output_norm(
            answer
            + F.dropout(
                self.feed_forward(answer), p=self.dropout, training=self.training
            )
        )
