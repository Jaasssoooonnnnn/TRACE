# Input caches

The selectors consume completed candidate trajectories and frozen embeddings. `trace-demo` creates small examples of both layouts. Tensor caches are loaded using `torch.load`; use caches from a trusted source.

To construct these caches from raw rollouts, see [preprocessing.md](preprocessing.md).

## WebQA

```text
webqa/
  train/
    tensor_graphs/part*.pt
    metadata/*.jsonl
    subquery_embeddings/*.pt
    chunk_embeddings/*.pt
  test/
    tensor_graphs/{nq,hotpotqa,triviaqa,popqa,2wiki,musique,bamboogle}.pt
    metadata/*.jsonl
    subquery_embeddings/*.pt
    chunk_embeddings/*.pt
```

Training and validation use a deterministic 95/5 split of normalized question text. The paper caches contain 101,323 training and 5,309 validation graphs.

Each tensor file contains `embedding_model`, `embedding_dim`, and `graphs`. A graph contains:

- `graph_id`, `question`, `split`, `trace_count`, and ordered `sample_ids`.
- `query_x` (1 × 4096) and `answer_x` (unique answers × 4096).
- `answer_em`, `answer_f1`, `answer_vote_counts`, and `answer_texts` in the same answer order.

Each metadata JSONL record is keyed by `graph_id` and contains:

- `trace_count`, `event_sample_ids`, and `event_answer_ids` (indices into `answer_x`).
- `event_subquery_text_ids`, `subquery_occurrence_to_event`, and `event_search_indices`.
- `event_to_event` for temporal links within each rollout.
- `event_chunk_content_ids` and `event_chunk_ranks`: three chunks, ranked 1/2/3, per event.
- `chunk_cross_rollout_coverage` in chunk-content order.

Subquery/chunk embedding files contain `schema_version="retrieval_text_embeddings_v1"`, `node_kind`, `embedding_model`, `embedding_dim`, `embeddings`, and `graphs`. Each entry of `graphs` contains `graph_id` and a LongTensor `embedding_indices` selecting that question's rows from the embedding table.

The default test denominator is 3,125 questions. For a different pool, add `manifest.json` at the data root:

```json
{"test": {"total_questions": 3125}}
```

## Long horizon

Place a `manifest.json` in each data root. Paths may be relative to the manifest directory or absolute. Training needs only `train` and `validation`; evaluation needs only `test`.

```json
{
  "train": {
    "graphs": "graphs.pt",
    "base_embeddings": "base_embeddings.pt",
    "subquery_embeddings": "subquery_embeddings.pt",
    "document_identities": "train_document_identities.json"
  },
  "validation": {
    "graphs": "graphs.pt",
    "base_embeddings": "base_embeddings.pt",
    "subquery_embeddings": "subquery_embeddings.pt",
    "document_identities": "validation_document_identities.json"
  },
  "test": {
    "graphs": "test_graphs.pt",
    "base_embeddings": "test_embeddings.pt",
    "subquery_embeddings": "test_embeddings.pt",
    "document_identities": "test_document_identities.json",
    "labels": "judgments.jsonl"
  }
}
```

The graph file contains `graphs`. Each graph has `qid`, `split`, `query_index`, and ordered `responses`. The training caches contain 2,655 training and 132 validation graphs. Each response contains:

- `seed`, `response_index`, `vote_count`, and boolean `correct` for training.
- `subqueries`: `subquery_id`, `query_index`, and `is_orphan`.
- `observations`: `event_id`, `tool_kind` (0=open, 1=find), `content_index`, `doc_index`, and `subquery_id`.
- `sample_id` for evaluation; if absent, it is `seed - 42`.

Embedding files contain `embedding_model`, `embedding_dim`, and `embeddings`. Query, response, and evidence indices refer to the base table; subquery indices refer to the subquery table. An orphan subquery has `query_index=-1`.

Document identity files contain a `questions` object keyed by string `qid`. Each question records `evidence_keys` (`[seed,event_id]`), `old_doc_indices`, `original_doc_indices`, `identities`, `evidence_identity_indices`, and `verified_mask`. Entries follow response/observation order. Verified identity keys are canonical `url:http://...`, `url:https://...`, or rollout-local `search-view:...`. Unresolved identities use `unresolved_original:<qid>:<doc_index>`.

Evaluation labels can be the existing `candidate_labels.json` object keyed by question key, with per-candidate `sample_id`, `correct`, and `candidate_hash`, or JSONL records:

```json
{"qid": 0, "sample_id": 0, "correct": true}
```

A test specification may include `cell` to select one cell from a combined cache. For FRAMES, add `"question_ids": "frames500_ids.json"` and copy the provided [ID list](frames500_ids.json) beside the manifest. The cached OpenResearcher FRAMES pool uses `"qid_offset": 10000`; GPT-OSS uses the default 0. Labels and document maps retain their original cache qids. The selected question set determines the evaluation denominator.
