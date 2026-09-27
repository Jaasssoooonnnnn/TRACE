"""Select existing candidates and score every question in the declared pool."""

import argparse
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from .engine import TASKS, configure_runtime, metrics, predict, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=TASKS)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    configure_runtime(args.task)
    model_class, dataset_class = TASKS[args.task]
    data = dataset_class(args.data, "test")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["task"] != args.task:
        raise ValueError("Checkpoint task mismatch")
    model = model_class(embedding_dim=data.embedding_dim).to(args.device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    loader = DataLoader(
        data, batch_size=256 if args.task == "webqa" else 4, shuffle=False
    )
    rows = predict(model, loader, args.device, args.task)
    for row in rows:
        graph = data.graphs[row["graph_index"]]
        if args.task == "webqa":
            row["graph_id"] = graph["graph_id"]
            canonical = data.answer_canonical_ids(graph)[row["candidate_index"]]
            if "answer_texts" in graph:
                row["answer"] = graph["answer_texts"][canonical]
        else:
            response = graph["responses"][row["candidate_index"]]
            row["sample_id"] = response.get("sample_id", int(response["seed"]) - 42)
            if "candidate_hash" in response:
                row["candidate_hash"] = response["candidate_hash"]
            if "answer" in response:
                row["answer"] = response["answer"]
    result = {
        "task": args.task,
        "epoch": checkpoint["epoch"],
        **metrics(rows, data.total_questions),
        "predictions": rows,
    }
    save_json(args.output, result)
    print({k: v for k, v in result.items() if k != "predictions"})


if __name__ == "__main__":
    main()
