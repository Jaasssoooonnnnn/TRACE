# TRACE

Trajectory Ranking with Aggregated Cross-Rollout Evidence.

This repository contains the WebQA and long-horizon selectors: graph construction from frozen trajectory caches, training, candidate selection, and evaluation.

## Install

Python 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
```

## Prepare trajectories and embeddings

Install the preprocessing dependencies with `python -m pip install -e '.[prepare]'`.
The three preprocessing commands parse completed rollouts, encode their text with
frozen Qwen3-Embedding-8B, and resolve browser document memberships:

```bash
trace-prepare webqa --input raw/nq_train.jsonl --dataset nq --split train \
  --output prepared/nq_train.jsonl
trace-prepare webqa --input raw/hotpotqa_train.jsonl --dataset hotpotqa --split train \
  --output prepared/hotpotqa_train.jsonl
trace-embed webqa --input prepared/nq_train.jsonl prepared/hotpotqa_train.jsonl \
  --output data/webqa

trace-prepare long_horizon --input raw/browser_train.jsonl --dataset browser --split train \
  --output prepared/browser_train.jsonl
trace-embed long_horizon --input prepared/browser_train.jsonl --output data/long_horizon
```

`--split train` applies the mixed-candidate training filter and the deterministic
question-level train/validation split. `--split test` retains every question,
including empty pools. Rollouts are selected by the first `--k` sample IDs before
validity filtering; invalid samples are not replaced. The default is `--k 16`.

`trace-embed long_horizon` also writes `document_identities.json`. To generate this
mapping separately, without loading the embedding model:

```bash
trace-documents --input prepared/browser_train.jsonl \
  --output prepared/document_identities.json
```

Raw input formats, provenance rules, and the embedding procedure are described in
[`data/preprocessing.md`](data/preprocessing.md).

## Train

```bash
trace-train webqa --data data/webqa --output runs/webqa
trace-train long_horizon --data data/long_horizon --output runs/long_horizon
```

Each run saves `best.pt`, selected by validation accuracy. WebQA breaks ties by validation F1; remaining ties use the earlier epoch.

The two configuration files are in [`trace_selector/configs`](trace_selector/configs). Text embeddings are frozen Qwen3-Embedding-8B vectors with 4096 dimensions. Both models use BCE, listwise ranking, and hardest-negative losses.

## Evaluate

```bash
trace-evaluate webqa --data data/webqa \
  --checkpoint runs/webqa/best.pt --output runs/webqa/test.json
trace-evaluate long_horizon --data data/browsecomp \
  --checkpoint runs/long_horizon/best.pt --output runs/long_horizon/browsecomp.json
```

Predictions identify the selected existing candidate by question and sample ID. Empty candidate pools count as incorrect. Dataset layout and judged-label formats are described in [`data/README.md`](data/README.md).

## CPU example

```bash
trace-demo --output data/demo
trace-train webqa --data data/demo/webqa --output runs/demo-webqa --device cpu
trace-train long_horizon --data data/demo/long_horizon --output runs/demo-long --device cpu
trace-evaluate webqa --data data/demo/webqa \
  --checkpoint runs/demo-webqa/best.pt --output runs/demo-webqa/test.json --device cpu
trace-evaluate long_horizon --data data/demo/long_horizon \
  --checkpoint runs/demo-long/best.pt --output runs/demo-long/test.json --device cpu
pytest -q
```

The example creates synthetic embeddings and labels to exercise the pipeline.
