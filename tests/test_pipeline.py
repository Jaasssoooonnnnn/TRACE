import json
import math
import shutil
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Batch

from trace_selector.data import question_split
from trace_selector.demo import make_long_horizon, make_webqa
from trace_selector.document import compile_question
from trace_selector.engine import TASKS, metrics
from trace_selector.losses import long_horizon_loss, webqa_loss
from trace_selector.train import run


@pytest.fixture()
def inputs(tmp_path):
    torch.manual_seed(42)
    make_webqa(tmp_path / "webqa")
    make_long_horizon(tmp_path / "long_horizon")
    return tmp_path


@pytest.mark.parametrize("task", TASKS)
def test_training_selects_only_validation(inputs, tmp_path, task):
    root = inputs / task
    if task == "webqa":
        shutil.rmtree(root / "test")
    else:
        path = root / "manifest.json"
        manifest = json.loads(path.read_text())
        del manifest["test"]
        path.write_text(json.dumps(manifest))
    config = json.loads(
        (Path(__file__).parents[1] / f"trace_selector/configs/{task}.json").read_text()
    )
    history = run(config, root, tmp_path / f"run-{task}", torch.device("cpu"))
    best = max(history, key=lambda r: (r["accuracy"], r.get("f1", 0), -r["epoch"]))
    checkpoint = torch.load(tmp_path / f"run-{task}/best.pt", weights_only=True)
    assert checkpoint["epoch"] == best["epoch"]
    assert len(list((tmp_path / f"run-{task}").glob("*.pt"))) == 1


@pytest.mark.parametrize("task", TASKS)
def test_batching_and_label_invariance(inputs, task):
    model_class, dataset_class = TASKS[task]
    dataset = dataset_class(inputs / task, "test")
    model = model_class().eval()
    items = [dataset[i] for i in range(len(dataset))]
    batch = Batch.from_data_list(items)
    with torch.no_grad():
        together = model(batch)
        separate = torch.cat([model(x) for x in items])
    torch.testing.assert_close(together, separate, atol=2e-5, rtol=2e-5)
    node, label = ("answer", "y_em") if task == "webqa" else ("response", "y")
    batch[node][label] = 1 - batch[node][label]
    if "y_f1" in batch[node]:
        batch[node].y_f1.zero_()
    with torch.no_grad():
        torch.testing.assert_close(model(batch), together, atol=0, rtol=0)


def test_loss_denominators():
    z = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
    y = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    groups = torch.tensor([0, 0, 1, 1, 2, 2])
    weight = torch.tensor(1.0)
    web = webqa_loss(z, y, pos_weight=weight, answer_batch=groups, num_graphs=3)
    long = long_horizon_loss(z, y, groups, 3, weight)
    torch.testing.assert_close(web, torch.tensor(2.5 * math.log(2)))
    torch.testing.assert_close(long, torch.tensor(3.0 * math.log(2)))
    (web + long).backward()
    assert torch.isfinite(z.grad).all()


def test_question_split_normalizes_unicode():
    assert question_split("  caf\u00e9  question ") == question_split(
        "cafe\u0301 question"
    )


def test_empty_questions_remain_in_denominator():
    assert metrics([{"correct": 1.0}], 2)["accuracy"] == 0.5
    assert metrics([], 2)["accuracy"] == 0.0


def test_search_view_never_links_rollouts():
    graph = {
        "qid": 1,
        "responses": [
            {"seed": seed, "observations": [{"event_id": 0, "doc_index": 0}]}
            for seed in [42, 43]
        ],
    }
    mapping = {
        "evidence_keys": [[42, 0], [43, 0]],
        "old_doc_indices": [0, 0],
        "original_doc_indices": [0],
        "identities": [{"key": "search-view:0", "verified": True}],
        "evidence_identity_indices": [0, 0],
        "verified_mask": [True, True],
    }
    with pytest.raises(AssertionError):
        compile_question(graph, mapping)


def test_missing_external_judgment_is_rejected(inputs):
    path = inputs / "long_horizon/labels.jsonl"
    path.write_text("\n".join(path.read_text().splitlines()[1:]) + "\n")
    with pytest.raises(KeyError):
        TASKS["long_horizon"][1](inputs / "long_horizon", "test")


def test_native_labels_and_frames_id_offset(inputs):
    root = inputs / "long_horizon"
    path = root / "graphs.pt"
    payload = torch.load(path, weights_only=False)
    mapping = json.loads((root / "document_identities.json").read_text())
    labels = {}
    for graph in payload["graphs"]:
        if graph["split"] != "test":
            continue
        old_qid = graph["qid"]
        graph["qid"] += 10000
        graph["key"] = f"frames/{old_qid:05d}"
        graph["cell"] = "frames"
        mapping["questions"][str(graph["qid"])] = mapping["questions"].pop(str(old_qid))
        rows = []
        for response in graph["responses"]:
            sid = response["sample_id"]
            response["candidate_hash"] = f"candidate-{old_qid}-{sid}"
            rows.append(
                {
                    "sample_id": sid,
                    "correct": sid == 1,
                    "candidate_hash": response["candidate_hash"],
                }
            )
        labels[graph["key"]] = rows
    torch.save(payload, path)
    (root / "document_identities.json").write_text(json.dumps(mapping))
    (root / "native_labels.json").write_text(json.dumps(labels))
    (root / "ids.json").write_text("[4]")
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["test"].update(
        labels="native_labels.json",
        question_ids="ids.json",
        qid_offset=10000,
        cell="frames",
    )
    (root / "manifest.json").write_text(json.dumps(manifest))
    dataset = TASKS["long_horizon"][1](root, "test")
    assert dataset.total_questions == 1
    assert dataset.graphs[0]["qid"] == 10004
    assert dataset.labels(0).tolist() == [0.0, 1.0, 0.0]
    labels["frames/00004"][0]["candidate_hash"] = "wrong"
    (root / "native_labels.json").write_text(json.dumps(labels))
    with pytest.raises(ValueError, match="hash mismatch"):
        TASKS["long_horizon"][1](root, "test")
