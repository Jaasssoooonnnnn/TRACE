"""Candidate BCE, listwise, and hard-ranking losses."""

import torch
from torch.nn import functional as F
from torch_geometric.utils import scatter


def webqa_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    pos_weight: torch.Tensor | None,
    answer_batch: torch.Tensor | None,
    num_graphs: int | None,
) -> torch.Tensor:
    if pos_weight is None:
        raise ValueError("BCE loss requires pos_weight")
    bce_loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
    if answer_batch is None or num_graphs is None:
        raise ValueError("listwise loss requires answer_batch and num_graphs")

    def grouped_logsumexp(values: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
        maxima = scatter(values, groups, dim=0, dim_size=num_graphs, reduce="max")
        sums = scatter(
            (values - maxima[groups]).exp(),
            groups,
            dim=0,
            dim_size=num_graphs,
            reduce="sum",
        )
        return maxima + sums.clamp_min(torch.finfo(values.dtype).tiny).log()

    positive = labels > 0.5
    all_logsumexp = grouped_logsumexp(logits, answer_batch)
    positive_logsumexp = grouped_logsumexp(logits[positive], answer_batch[positive])
    positive_graph = scatter(
        positive.float(), answer_batch, dim=0, dim_size=num_graphs, reduce="sum"
    ).gt(0)
    losses = all_logsumexp - positive_logsumexp
    weights = positive_graph.to(losses.dtype)
    listwise_loss = (losses * weights).sum() / weights.sum().clamp_min(1.0)
    loss = bce_loss + listwise_loss
    negative = ~positive
    negative_graph = scatter(
        negative.float(), answer_batch, dim=0, dim_size=num_graphs, reduce="sum"
    ).gt(0)
    valid_hard_graph = positive_graph & negative_graph
    max_positive = scatter(
        logits[positive],
        answer_batch[positive],
        dim=0,
        dim_size=num_graphs,
        reduce="max",
    )
    max_negative = scatter(
        logits[negative],
        answer_batch[negative],
        dim=0,
        dim_size=num_graphs,
        reduce="max",
    )
    max_positive = torch.where(positive_graph, max_positive, 0.0)
    max_negative = torch.where(negative_graph, max_negative, 0.0)
    hard_losses = F.softplus(max_negative - max_positive)
    hard_weights = valid_hard_graph.to(hard_losses.dtype)
    hard_loss = (hard_losses * hard_weights).sum() / hard_weights.sum().clamp_min(1.0)
    return loss + hard_loss


def grouped_logsumexp(
    values: torch.Tensor, groups: torch.Tensor, graph_count: int
) -> torch.Tensor:
    maxima = scatter(values, groups, dim=0, dim_size=graph_count, reduce="max")
    sums = scatter(
        (values - maxima[groups]).exp(),
        groups,
        dim=0,
        dim_size=graph_count,
        reduce="sum",
    )
    return maxima + sums.clamp_min(torch.finfo(values.dtype).tiny).log()


def long_horizon_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    answer_batch: torch.Tensor,
    graph_count: int,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
    positive = labels > 0.5
    positive_counts = scatter(
        positive.long(), answer_batch, dim=0, dim_size=graph_count, reduce="sum"
    )
    negative_counts = scatter(
        (~positive).long(), answer_batch, dim=0, dim_size=graph_count, reduce="sum"
    )
    valid = (positive_counts > 0) & (negative_counts > 0)
    if not bool(valid.any()):
        return bce
    all_lse = grouped_logsumexp(logits, answer_batch, graph_count)
    positive_lse = grouped_logsumexp(
        logits[positive], answer_batch[positive], graph_count
    )
    listwise = (all_lse - positive_lse)[valid].mean()
    negative = ~positive
    max_positive = scatter(
        logits[positive],
        answer_batch[positive],
        dim=0,
        dim_size=graph_count,
        reduce="max",
    )
    max_negative = scatter(
        logits[negative],
        answer_batch[negative],
        dim=0,
        dim_size=graph_count,
        reduce="max",
    )
    hard = F.softplus(max_negative - max_positive)[valid].mean()
    return bce + listwise + hard
