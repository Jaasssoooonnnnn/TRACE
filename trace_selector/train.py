"""Train TRACE selectors on frozen text embeddings."""

import argparse
import json
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from .engine import (
    TASKS,
    configure_runtime,
    metrics,
    positive_weight,
    predict,
    save_json,
    seed_all,
    train_epoch,
)


def run(config, data_dir, output, device):
    task = config["task"]
    configure_runtime(task)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {output}")
    model_class, dataset_class = TASKS[task]
    if task == "webqa":
        dataset = dataset_class(data_dir, None)
        train, validation = dataset.subset("train"), dataset.subset("validation")
    else:
        train = dataset_class(data_dir, "train")
        validation = dataset_class(data_dir, "validation")
    if not train or not validation:
        raise ValueError("Training and validation must both contain questions")
    seed_all(42)
    train_loader = DataLoader(
        train, batch_size=config["batch_size"], shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        validation, batch_size=config["batch_size"], shuffle=False, num_workers=0
    )
    model = model_class(embedding_dim=train.embedding_dim).to(device)
    kwargs = {"foreach": False} if task == "long_horizon" else {}
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        **kwargs,
    )
    weight = positive_weight(train, device)
    save_json(
        output / "config.json",
        {
            **config,
            "positive_weight": float(weight),
            "train_questions": len(train),
            "validation_questions": len(validation),
        },
    )
    history = []
    best = None
    for epoch in range(1, config["epochs"] + 1):
        if task == "long_horizon":
            train_loader = DataLoader(
                train, batch_size=config["batch_size"], shuffle=True, num_workers=0
            )
        loss = train_epoch(model, train_loader, optimizer, device, weight, task)
        scores = metrics(
            predict(model, val_loader, device, task), validation.total_questions
        )
        row = {"epoch": epoch, "train_loss": loss, **scores}
        history.append(row)
        key = (scores["accuracy"], scores.get("f1", 0), -epoch)
        if best is None or key > best[0]:
            best = (key, epoch)
            torch.save(
                {
                    "task": task,
                    "epoch": epoch,
                    "embedding_dim": train.embedding_dim,
                    "model_state": model.state_dict(),
                },
                output / "best.pt",
            )
        save_json(output / "history.json", history)
        print(json.dumps(row), flush=True)
    save_json(
        output / "selection.json",
        {
            "validation_epoch": best[1],
            "validation_rule": "accuracy, then F1 (WebQA only), then earlier epoch",
        },
    )
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=TASKS)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    config = json.loads(
        (Path(__file__).parent / "configs" / f"{args.task}.json").read_text()
    )
    run(config, args.data, args.output, torch.device(args.device))


if __name__ == "__main__":
    main()
