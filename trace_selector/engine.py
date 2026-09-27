"""Shared training and evaluation utilities."""

import json
import random
from pathlib import Path

import numpy as np
import torch

from .long_horizon_data import LongHorizonDataset
from .long_horizon_model import LongHorizonSelector
from .losses import long_horizon_loss, webqa_loss
from .webqa_data import WebQADataset
from .webqa_model import WebQASelector

TASKS = {
    "webqa": (WebQASelector, WebQADataset),
    "long_horizon": (LongHorizonSelector, LongHorizonDataset),
}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_runtime(task):
    torch.set_num_threads(4)
    if task == "long_horizon":
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def positive_weight(dataset, device):
    positive = total = 0
    for index in range(len(dataset)):
        y = dataset.labels(index)
        positive += float(y.sum())
        total += y.numel()
    return torch.tensor((total - positive) / max(positive, 1), device=device)


def train_epoch(model, loader, optimizer, device, weight, task):
    model.train()
    loss_sum = count = 0
    node, label = ("answer", "y_em") if task == "webqa" else ("response", "y")
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(data)
        if task == "webqa":
            loss = webqa_loss(
                logits,
                data[node][label],
                pos_weight=weight,
                answer_batch=data[node].batch,
                num_graphs=data.num_graphs,
            )
        else:
            loss = long_horizon_loss(
                logits, data[node][label], data[node].batch, data.num_graphs, weight
            )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Nonfinite training loss")
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.detach()) * data.num_graphs
        count += data.num_graphs
    if count == 0:
        raise ValueError("Training split is empty")
    return loss_sum / count


@torch.inference_mode()
def predict(model, loader, device, task):
    model.eval()
    node, label = ("answer", "y_em") if task == "webqa" else ("response", "y")
    rows = []
    for data in loader:
        data = data.to(device)
        scores = model(data)
        if not bool(torch.isfinite(scores).all()):
            raise RuntimeError("Nonfinite prediction score")
        ptr = data[node].ptr.tolist()
        for i, (start, end) in enumerate(zip(ptr, ptr[1:])):
            if start == end:
                raise ValueError("Empty candidate graph must be counted separately")
            selected = start + int(scores[start:end].argmax())
            row = {
                "graph_index": int(data.graph_index[i]),
                "candidate_index": selected - start,
                "score": float(scores[selected]),
            }
            if label in data[node]:
                row["correct"] = float(data[node][label][selected])
            if task == "webqa":
                row["sample_id"] = int(data[node].sample_id[selected])
                if "y_f1" in data[node]:
                    row["f1"] = float(data[node].y_f1[selected])
            else:
                row["qid"] = int(data.qid[i])
                row["seed"] = int(data[node].seed[selected])
            rows.append(row)
    return rows


def metrics(rows, total_questions):
    if len(rows) > total_questions or total_questions <= 0:
        raise ValueError("Invalid evaluation denominator")
    result = {
        "total_questions": total_questions,
        "selected_questions": len(rows),
        "empty_questions": total_questions - len(rows),
    }
    if all("correct" in row for row in rows):
        result["correct"] = sum(r["correct"] for r in rows)
        result["accuracy"] = result["correct"] / total_questions
    if rows and all("f1" in row for row in rows):
        result["f1"] = sum(r["f1"] for r in rows) / total_questions
    return result
