# Trajectory preprocessing

The preprocessing commands convert completed rollouts into TRACE input caches.
Inputs may be JSONL or Parquet; each record represents one rollout.

## WebQA input

```json
{"question_index": 0, "question": "What is the capital of France?", "golden_answers": ["Paris"], "sample_id": 0, "answer": "Paris", "steps": [{"action": "search", "content": "capital of France", "search_result": "Doc 1 (Title: France) Paris is the capital.\nDoc 2 (Title: Paris) Paris is a city.\nDoc 3 (Title: Geography) France is in Europe."}]}
```

`question_index` groups rollouts, and `sample_id` orders them. A nonempty Search
return must contain `Doc 1`, `Doc 2`, and `Doc 3` in that order. The rank marker is
removed from the embedding text; the title and content are retained. Chunk
identity applies Unicode NFC normalization and collapses whitespace. Malformed
nonempty returns raise an error.

The parser keeps rollouts with a nonempty final answer and at least one nonempty
Search return. Every retained Search and chunk remains a separate occurrence,
including repetitions within a rollout. Metadata records the original step,
sample, Search order, chunk rank, answer membership, and temporal predecessor.
Text embeddings are shared across occurrences with identical content.

For training, the mixed-answer filter is applied to the first K original samples
before validity filtering. WebQA EM/F1 lowercase text, remove ASCII punctuation
and English articles, and compare
against the best reference alias. Empty normalized predictions score zero.
Answer-frequency features are computed over retained candidates.

Prepare each dataset separately, then encode the parsed files together:

```bash
trace-prepare webqa --input raw/nq_test.jsonl --dataset nq --split test --output prepared/nq_test.jsonl
trace-prepare webqa --input raw/hotpotqa_test.jsonl --dataset hotpotqa --split test --output prepared/hotpotqa_test.jsonl
trace-embed webqa --input prepared/nq_test.jsonl prepared/hotpotqa_test.jsonl --output data/webqa_test
```

Dataset names are `nq`, `hotpotqa`, `triviaqa`, `popqa`, `2wiki`, `musique`, and
`bamboogle`. Pass all seven test inputs to reproduce the complete evaluation
pool. The evaluation denominator is the number of input questions.
Training tensors are written in shards; `--shard-size` defaults to 512 questions.
The resulting directory can be passed directly to `trace-train` or
`trace-evaluate`. Use an empty cache output directory.

## Browser input

Raw browser records contain `qid`, `question`, `sample_id` (or a zero-based
`traj_idx`), a saved boolean `correct` judgment, and `messages`:

```json
{"qid": 0, "question": "What is the capital of France?", "sample_id": 0, "correct": true, "messages": [{"role": "assistant", "recipient": "browser.search", "content": "{\"query\":\"capital of France\"}"}, {"role": "tool", "name": "browser.search", "content": "[0] Results (web-search://capital)"}, {"role": "assistant", "recipient": "browser.open", "content": "{\"cursor\":0,\"id\":0}"}, {"role": "tool", "name": "browser.open", "content": "[1] Doc 10 (https://example.org/france)\n**viewing lines [1 - 2] of 2**\nL1: URL: https://example.org/france\nL2: Paris is the capital of France."}, {"role": "assistant", "channel": "final", "content": "Paris"}]}
```

Native `recipient`/tool-name messages and OpenAI-style `tool_calls`/`tool_call_id`
pairs are supported. Tool names are `browser.search`, `browser.open`, and
`browser.find`. If `seed` is present and `sample_id` is absent, the sample ID is
`seed - 42`; otherwise a missing seed is set to `sample_id + 42`. Questions are
kept under their supplied qids. For combined datasets, qids must be unique.

The last assistant final message supplies the candidate answer. For chat logs
without channel annotations, the last assistant message without a tool call
supplies it. The optional `candidate_answer` field accepts an answer extracted
from the model response without reference answers.
Votes group candidates by the same answer normalization as WebQA.

If judgments live in a separate file, pass `--labels judgments.jsonl`, with
records in the existing `qid`, `sample_id`, boolean `correct` format for training
and evaluation.

The parser also accepts typed metadata: `qid`, `seed`,
`correct`, `query_content_hash`, `candidate_answer_content_hash`, and
`tool_events`. Events carry `event_id`, `message_index`, `tool_type`,
`arguments`, `arguments_valid`, `source_event_id`, `source_cursor`,
`result_cursor`, `result_message_index`, and `observation_parts` with
`kind`/`content_hash`. Pass `--corpus corpus.jsonl` to resolve hashes from records
containing `text_hash` and `text`. Corpus text hashes are validated. Inline
question, answer, result, and observation text fields can replace hash lookups.

### Filtering and provenance

Only the first K samples are considered; invalid samples are not replaced.
Browsing retains nonempty candidate answers, including candidates with no
observations. Training retains questions with both correct and incorrect
retained candidates. Test preparation preserves empty questions.

Each Search creates a Subquery occurrence. Open observations have their cursor,
viewport markers, and line-number prefixes removed before whitespace
normalization. Find observations concatenate nonempty match excerpts with two
newlines, preserving one Evidence occurrence per Find event. Failed Open calls
and zero-match Find calls do not create Evidence nodes.

For each tool call, the parser resolves the supplied cursor to a previously
returned view. A cursor of `-1`, including the tool's omitted-cursor default,
means the most recently returned view. Following the parent-event chain
identifies its Search ancestor. A chain without a Search ancestor gets
an orphan Subquery with a zero embedding; later observations in that chain
share its orphan context. Missing Search query text also uses a zero embedding.
All Search nodes remain in temporal order, including Searches whose results are
not subsequently opened.

Supplied typed provenance is checked against call/return order. Duplicate
returned cursors, contradictory source annotations, invalid arguments, or
missing returns cannot establish verified source identities.

### Verified document identity

```bash
trace-prepare long_horizon --input raw/frames.jsonl --labels raw/judgments.jsonl \
  --dataset frames --split test --output prepared/frames.jsonl
trace-documents --input prepared/frames.jsonl --output prepared/document_identities.json
trace-embed long_horizon --input prepared/frames.jsonl --output data/frames
```

An Open's returned page header establishes its source. A corroborating encoded
`URL:` field takes precedence over its decoded display. Redirects use the
returned destination. Find and page-continuation Open inherit their verified
parent's source. A
returned URL that conflicts with that source leaves the observation unresolved.

URL identities lowercase scheme and authority, remove fragments, normalize
percent escapes for unreserved characters, and preserve query information and
encoded reserved delimiters: `%2F` remains distinct from `/`, and `%26` from `&`.
URL userinfo is represented by an opaque hash.

A bare `Doc N` header can use a same-question Doc-ID mapping only when complete
returned headers establish one unambiguous URL for that ID. Malformed or
truncated URL headers remain unresolved.

Search-result pages have rollout-local `search-view:` identities. Unverified
observations use `unresolved_original:<qid>:<doc_index>`, separated from all
verified URL groups. Local document groups follow navigation chains and explicit
URL targets; the verified map supplies the final shared memberships.
`verified` records whether returned-page evidence and navigation provenance
establish the source identity.

The generated mapping records `evidence_keys`, `old_doc_indices`,
`original_doc_indices`, `identities`, `evidence_identity_indices`,
`verified_mask`, and resolution statuses in exact response/observation order.
`trace-embed long_horizon` generates the same mapping alongside the tensors,
saved judgments, and loader manifest.

## Frozen Qwen3 embedding procedure

`trace-embed` loads `Qwen/Qwen3-Embedding-8B` in evaluation mode with gradients
disabled. It encodes the prepared question, Subquery, Evidence, and candidate
answer text without an instruction prefix or chat template.

Tokenization uses left padding and right truncation at 4,096 tokens, including
the tokenizer's special tokens. The final non-padding hidden state is sliced
to 4,096 dimensions, converted to float32, and L2 normalized. By default,
inference uses BF16 and normalized vectors are stored in FP16. Use
`--save-float32` to retain float32 storage. `--batch-size` defaults to 16;
`--max-length` changes the truncation limit and is recorded in cache metadata.

`--model-path` can point to a local copy of the same Qwen3-Embedding-8B weights.
Cache metadata records the public model ID. CUDA is selected when available;
`--device cpu --dtype float32` is also supported.

Exact text reuse occurs within each encoded table, with separate graph nodes
for each occurrence. Orphan Subqueries use zero vectors. WebQA writes rank and
provenance sidecars; browsing writes base and Subquery embedding tables
compatible with the existing loaders.
