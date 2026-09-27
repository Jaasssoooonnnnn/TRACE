import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from trace_selector.embeddings import (
    encode_texts,
    last_token_pool,
    write_long_horizon,
    write_webqa,
)
from trace_selector.engine import TASKS, metrics, predict
from trace_selector.prepare import prepare_pools
from trace_selector.source_identity import (
    build_document_identities,
    canonical_url,
    event_identities,
    page_identity,
)
from trace_selector.trajectories import (
    browser_events,
    build_browser_pool,
    build_webqa_pool,
    ranked_chunks,
)


def webqa_rows():
    result = "\n".join(
        f"Doc {rank} (Title: Source) shared fact {rank}" for rank in (1, 2, 3)
    )
    return [
        {
            "question_index": 0,
            "question": "A test question?",
            "golden_answers": ["Paris"],
            "sample_id": i,
            "answer": answer,
            "steps": [
                {"action": "search", "content": "where", "search_result": result}
            ],
        }
        for i, answer in enumerate(["Paris", "London", "", "Paris"])
    ]


def browsing_row(sample=0):
    messages = []

    def tool(kind, args, text):
        messages.extend(
            [
                {
                    "role": "assistant",
                    "recipient": "browser." + kind,
                    "content": json.dumps(args),
                },
                {"role": "tool", "name": "browser." + kind, "content": text},
            ]
        )

    tool("search", {"query": "first"}, "[0] First (web-search://first)")
    tool(
        "open",
        {"cursor": 0, "id": 0},
        "[1] Doc 10 (https://example.org/a)\n**viewing lines [1 - 2] of 2**\nL1: URL: https://example.org/a\nL2: A fact",
    )
    tool("search", {"query": "second"}, "[2] Second (web-search://second)")
    tool(
        "find",
        {"cursor": 1, "pattern": "fact"},
        "[3] Find results for text: fact in Doc 10\nL1: # 【0†match at L2】\nL2: A fact",
    )
    tool(
        "open",
        {"cursor": 999, "id": "https://example.org/b"},
        "[4] Doc 11 (https://example.org/b)\n**viewing lines [1 - 2] of 3**\nL1: URL: https://example.org/b\nL2: B fact",
    )
    tool(
        "open",
        {"cursor": 4, "id": -1},
        "[5] Doc 11 (https://example.org/b)\n**viewing lines [3 - 3] of 3**\nL3: Another B fact",
    )
    tool("find", {"cursor": 1, "pattern": "absent"}, "[6] No matches found")
    tool("open", {"cursor": 0, "id": 99}, "Error: no such link")
    messages.append(
        {
            "role": "assistant",
            "channel": "final",
            "content": "Paris" if sample == 0 else "London",
        }
    )
    return {
        "qid": 0,
        "question": "A browser question?",
        "sample_id": sample,
        "correct": sample == 0,
        "messages": messages,
    }


class SyntheticEncoder:
    def metadata(self):
        return {"embedding_model": "synthetic-demo", "embedding_dim": 4096}

    def __call__(self, texts):
        vectors = torch.zeros(len(texts), 4096)
        for i, text in enumerate(texts):
            position = (
                int.from_bytes(hashlib.sha256(text.encode()).digest()[:2], "big") % 4096
            )
            vectors[i, position] = 1
        return vectors


def test_webqa_filtering_and_occurrence_membership():
    pool = build_webqa_pool(webqa_rows(), "nq", "test", 3)
    assert pool["sample_ids"] == [0, 1]  # Sample 3 never replaces invalid sample 2.
    assert pool["trace_count"] == 2
    assert pool["metadata"]["event_answer_ids"] == [0, 1]
    assert pool["metadata"]["chunk_cross_rollout_coverage"] == [2, 2, 2]
    repeated = webqa_rows()[:1]
    repeated[0]["steps"] *= 2
    pool = build_webqa_pool(repeated, "nq", "test", 16)
    assert pool["metadata"]["event_to_event"] == [[0, 1]]
    assert pool["metadata"]["subquery_occurrence_to_event"] == [[0, 0], [1, 1]]
    assert len(pool["subquery_texts"]) == 1


def test_webqa_training_filter_and_rank_validation():
    assert build_webqa_pool(webqa_rows()[:1], "nq", "train", 16) is None
    assert build_webqa_pool(webqa_rows()[:2], "nq", "train", 16) is not None
    chunks = ranked_chunks(
        "Doc 1 (Title: café) A\nDoc 2 (Title: B) B\nDoc 3 (Title: C) C"
    )
    assert chunks[0] == "(Title: café) A"
    with pytest.raises(ValueError, match="ranks"):
        ranked_chunks("Doc 2 (Title: A) A\nDoc 1 (Title: B) B\nDoc 3 (Title: C) C")


def test_browser_provenance_and_document_inheritance():
    pool = build_browser_pool([browsing_row()], "frames", "test", 16)
    response = pool["responses"][0]
    assert [o["event_id"] for o in response["observations"]] == [1, 3, 4, 5]
    assert [o["subquery_id"] for o in response["observations"]] == [
        "search:0",
        "search:0",
        "orphan:4",
        "orphan:4",
    ]
    assert [s["subquery_id"] for s in response["subqueries"]] == [
        "search:0",
        "search:2",
        "orphan:4",
    ]
    mapping = build_document_identities([pool])["questions"]["0"]
    assert mapping["verified_mask"] == [True] * 4
    assert mapping["evidence_identity_indices"] == [0, 0, 1, 1]
    assert [i["key"] for i in mapping["identities"]] == [
        "url:https://example.org/a",
        "url:https://example.org/b",
    ]


def test_identity_is_independent_of_answers_and_labels():
    pool = build_browser_pool([browsing_row(0), browsing_row(1)], "frames", "test", 16)
    before = build_document_identities([pool])
    changed = copy.deepcopy(pool)
    for response in changed["responses"]:
        response["answer"] = "Unrelated answer"
        response["correct"] = not response["correct"]
        response["vote_count"] = 999
    assert build_document_identities([changed]) == before


def test_native_gold_answer_never_supplies_the_candidate():
    row = browsing_row(1)
    row["answer"] = "Paris"  # The native Parquet answer column is a reference.
    row["messages"][-1]["recipient"] = "all"
    row["messages"][-1]["content"] = [{"text": None}, {"text": "London"}]
    pool = build_browser_pool([row], "frames", "test", 16)
    assert pool["responses"][0]["answer"] == "London"


def test_missing_search_text_uses_zero_embeddings(tmp_path):
    row = browsing_row()
    row["messages"][0]["content"] = '{"query":""}'
    pool = build_browser_pool([row], "frames", "test", 16)
    root = tmp_path / "cache"
    root.mkdir()
    write_long_horizon([pool], root, SyntheticEncoder())
    dataset = TASKS["long_horizon"][1](root, "test")
    data = dataset[0]
    assert data["subquery"].x[0].count_nonzero() == 0
    assert not data["subquery"].is_orphan[0]


def test_conflicting_cursor_is_not_verified():
    row = browsing_row()
    # A reused returned cursor makes both owners ambiguous.
    row["messages"][5]["content"] = "[1] Second (web-search://second)"
    pool = build_browser_pool([row], "frames", "test", 16)
    mapping = build_document_identities([pool])["questions"]["0"]
    assert mapping["verified_mask"][:2] == [False, False]
    assert all(
        not i["key"].startswith("url:")
        for i in mapping["identities"]
        if not i["verified"]
    )


def test_search_views_stay_local_and_scrolling_preserves_source():
    messages = [
        {
            "role": "assistant",
            "recipient": "browser.search",
            "content": '{"query":"q"}',
        },
        {"role": "tool", "name": "browser.search", "content": "[0] Q (web-search://q)"},
        {
            "role": "assistant",
            "recipient": "browser.open",
            "content": '{"cursor":0,"id":-1}',
        },
        {"role": "tool", "name": "browser.open", "content": "[1] Q (web-search://q)"},
    ]
    events = browser_events(messages)
    left = event_identities(events, "trajectory-a")
    right = event_identities(events, "trajectory-b")
    assert left[0]["key"] == left[1]["key"]
    assert left[1]["key"] != right[1]["key"]


def test_url_identity_preserves_reserved_delimiters():
    assert (
        canonical_url("HTTPS://EXAMPLE.org/%7euser/a%2fb?x=a%26b#part")
        == "https://example.org/~user/a%2Fb?x=a%26b"
    )
    assert canonical_url("https://example.org/a%2Fb") != canonical_url(
        "https://example.org/a/b"
    )
    assert canonical_url("https://example.org/?x=a%26b") != canonical_url(
        "https://example.org/?x=a&b"
    )
    assert canonical_url("https://example.org/a%ZZ") is None
    private = canonical_url("https://user:password@example.org/a")
    assert "password" not in private and "userinfo-sha256-" in private


def test_returned_display_is_corroborated_by_encoded_url_field():
    proof = page_identity(
        "Doc 4 (https://example.org/a/b?x=a&b)\n**viewing lines [1 - 2] of 2**\nL1: URL: https://example.org/a%2Fb?x=a%26b\nL2: Page body"
    )
    assert proof["url"] == "https://example.org/a%2Fb?x=a%26b"
    assert proof["field_verified"]
    assert page_identity("Doc 4 (https://example.org/...)\nL1: Body")["url"] is None
    assert (
        page_identity("Unknown header\nL1: Body says (https://example.org/a)")["url"]
        is None
    )


def test_pooling_and_encoding_truncation():
    hidden = torch.arange(2 * 4 * 3).reshape(2, 4, 3)
    mask = torch.tensor([[0, 0, 1, 1], [1, 1, 0, 0]])
    torch.testing.assert_close(
        last_token_pool(hidden, mask), torch.stack([hidden[0, 3], hidden[1, 1]])
    )
    calls = []

    def tokenizer(texts, **kwargs):
        calls.append((texts, kwargs))
        return {
            "input_ids": torch.tensor([[1, 2]] * len(texts)),
            "attention_mask": torch.ones(len(texts), 2, dtype=torch.long),
        }

    def model(**tokens):
        assert not torch.is_grad_enabled()
        return SimpleNamespace(
            last_hidden_state=torch.ones(tokens["input_ids"].shape[0], 2, 4096)
        )

    vectors = encode_texts(
        ["raw q", "raw evidence"], tokenizer, model, "cpu", batch_size=1, max_length=7
    )
    assert [c[0] for c in calls] == [["raw q"], ["raw evidence"]]
    assert all(c[1]["max_length"] == 7 and c[1]["truncation"] for c in calls)
    torch.testing.assert_close(vectors.norm(dim=1), torch.ones(2))
    assert vectors.dtype == torch.float32


@pytest.mark.parametrize("task", ["webqa", "long_horizon"])
def test_prepared_caches_work_with_existing_selectors(tmp_path, task):
    root = tmp_path / task
    root.mkdir()
    if task == "webqa":
        raw = webqa_rows()[:2]
        empty = copy.deepcopy(raw[0])
        empty.update(question_index=1, question="An empty question?", answer="")
        pools = list(prepare_pools(raw + [empty], task, "nq", "test", 16))
        write_webqa(pools, root, SyntheticEncoder())
    else:
        raw = [browsing_row(0), browsing_row(1)]
        empty = copy.deepcopy(raw[0])
        empty.update(qid=1, question="An empty browser question?")
        empty["messages"][-1]["content"] = ""
        pools = list(prepare_pools(raw + [empty], task, "frames", "test", 16))
        write_long_horizon(pools, root, SyntheticEncoder())
    model_class, dataset_class = TASKS[task]
    dataset = dataset_class(root, "test")
    assert dataset.total_questions == 2 and len(dataset) == 1
    data = dataset[0]
    if task == "webqa":
        assert data["chunk_occurrence"].num_nodes == 6
        assert data[
            "chunk_occurrence", "same_content_cross_trace", "chunk_occurrence"
        ].edge_index.shape == (2, 6)
    else:
        assert data["evidence"].num_nodes == 8
        assert data["document"].num_nodes >= 2
    from torch_geometric.loader import DataLoader

    rows = predict(model_class().eval(), DataLoader(dataset, batch_size=2), "cpu", task)
    assert metrics(rows, 2)["empty_questions"] == 1
