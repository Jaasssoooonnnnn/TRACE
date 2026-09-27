"""Verified source identity with isolated unresolved document groups."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def compile_question(graph, question):
    keys = [
        [int(r["seed"]), int(e["event_id"])]
        for r in graph["responses"]
        for e in r["observations"]
    ]
    old = [int(e["doc_index"]) for r in graph["responses"] for e in r["observations"]]
    assert question["evidence_keys"] == keys and question["old_doc_indices"] == old
    assert question["original_doc_indices"] == sorted(set(old))
    identities = question["identities"]
    memberships = question["evidence_identity_indices"]
    mask = question["verified_mask"]
    assert len(memberships) == len(old) == len(mask) and set(memberships) == set(
        range(len(identities))
    )
    for i, identity in enumerate(identities):
        indices = [j for j, g in enumerate(memberships) if g == i]
        name = identity["key"]
        assert all((mask[j] == identity["verified"] for j in indices))
        if identity["verified"]:
            assert name.startswith(("url:http://", "url:https://", "search-view:"))
            if name.startswith("search-view:"):
                assert len({keys[j][0] for j in indices}) == 1
        else:
            old_ids = {old[j] for j in indices}
            assert len(old_ids) == 1
            assert name == f"unresolved_original:{graph['qid']}:{next(iter(old_ids))}"
    original = {doc: i for i, doc in enumerate(sorted(set(old)))}
    capacity = max(len(original), len(identities))
    weights = np.zeros((len(identities), capacity), dtype=np.int64)
    for group, doc in zip(memberships, old):
        weights[group, original[doc]] += 1
    slots = {}
    if identities:
        rows, columns = linear_sum_assignment(weights, maximize=True)
        slots = dict(zip(rows.tolist(), columns.tolist()))
    endpoints = [slots[group] for group in memberships]
    return dict(
        qid=int(graph["qid"]),
        capacity=capacity,
        old_capacity=len(original),
        identities=len(identities),
        evidence=len(old),
        slots=endpoints,
        preserved_endpoints=sum(
            (slot == original[doc] for slot, doc in zip(endpoints, old))
        ),
    )
