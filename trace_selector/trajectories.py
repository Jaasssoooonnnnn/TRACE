"""Parse completed WebQA and browser rollouts without merging occurrences."""

import json
import re
import string
import unicodedata
from collections import Counter, defaultdict

from .data import question_split
from .source_identity import canonical_url

BROWSER_TOOLS = {"browser.search", "browser.open", "browser.find"}
DOC_START = re.compile(r"(?m)^[ \t]*Doc\s+(\d+)\s*(?=\(Title:)")


def normalize_text(value):
    return " ".join(unicodedata.normalize("NFC", str(value or "")).split())


def normalize_answer(value):
    text = str(value or "").lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def answer_scores(answer, references):
    prediction = normalize_answer(answer)
    em = float(
        bool(prediction) and any(prediction == normalize_answer(x) for x in references)
    )
    tokens = Counter(prediction.split())
    f1 = 0.0
    for reference in references:
        target = Counter(normalize_answer(reference).split())
        overlap = sum((tokens & target).values())
        if overlap:
            f1 = max(f1, 2 * overlap / (sum(tokens.values()) + sum(target.values())))
    return em, f1


def ranked_chunks(value):
    raw = (
        unicodedata.normalize("NFC", str(value or ""))
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .strip()
    )
    matches = list(DOC_START.finditer(raw))
    if not matches or raw[: matches[0].start()].strip():
        raise ValueError("Nonempty Search return must begin with Doc 1 (Title:...)")
    if [int(match[1]) for match in matches] != [1, 2, 3]:
        raise ValueError("Search return must contain ranks 1, 2, 3 in order")
    chunks = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        text = normalize_text(raw[match.end() : end])
        if not text:
            raise ValueError("Empty ranked chunk")
        chunks.append(text)
    return chunks


def _first_k(records, k, sample_key):
    by_sample = {}
    for record in records:
        sample = sample_key(record)
        if sample in by_sample:
            raise ValueError(f"Duplicate rollout sample ID: {sample}")
        by_sample[sample] = record
    return [by_sample[s] for s in sorted(by_sample)[:k]]


def build_webqa_pool(records, dataset, split, k):
    first = records[0]
    qid = str(first.get("question_id", first["question_index"]))
    question = str(first["question"])
    references = first["golden_answers"]
    references = references if isinstance(references, list) else [str(references)]
    selected = _first_k(records, k, lambda r: int(r["sample_id"]))
    if split == "train":
        labels = {answer_scores(r.get("answer"), references)[0] for r in selected}
        if labels != {0.0, 1.0}:
            return None
    kept = []
    for record in selected:
        if (
            str(record["question"]) != question
            or record["golden_answers"] != first["golden_answers"]
        ):
            raise ValueError("Conflicting question or references within one pool")
        events = []
        for step_index, step in enumerate(record.get("steps") or []):
            if not isinstance(step, dict) or not normalize_text(
                step.get("search_result")
            ):
                continue
            if step.get("action") != "search":
                raise ValueError("Retrieval output belongs to a non-Search step")
            events.append(
                (
                    step_index,
                    normalize_text(step.get("content")),
                    ranked_chunks(step["search_result"]),
                )
            )
        if str(record.get("answer") or "").strip() and events:
            kept.append((record, events))
    if split == "train" and not kept:
        return None
    answer_texts, subquery_texts, chunk_texts = [], [], []
    answer_index, subquery_index, chunk_index = {}, {}, {}

    def intern(text, table, index):
        if text not in index:
            index[text] = len(table)
            table.append(text)
        return index[text]

    metadata = {
        field: []
        for field in (
            "event_sample_ids",
            "event_answer_ids",
            "event_subquery_text_ids",
            "event_search_indices",
            "event_raw_step_indices",
            "event_chunk_content_ids",
            "event_chunk_ranks",
            "subquery_occurrence_to_event",
            "event_to_event",
        )
    }
    votes = Counter()
    coverage = defaultdict(set)
    subquery_occurrences = 0
    for record, events in kept:
        sample = int(record["sample_id"])
        answer = str(record["answer"]).strip()
        key = normalize_answer(answer)
        if key not in answer_index:
            answer_index[key] = len(answer_texts)
            answer_texts.append(answer)
        answer_id = answer_index[key]
        votes[answer_id] += 1
        previous = None
        for search_index, (step_index, query, chunks) in enumerate(events):
            event = len(metadata["event_sample_ids"])
            query_id = intern(query, subquery_texts, subquery_index) if query else -1
            ids = [intern(text, chunk_texts, chunk_index) for text in chunks]
            for chunk in ids:
                coverage[chunk].add(sample)
            metadata["event_sample_ids"].append(sample)
            metadata["event_answer_ids"].append(answer_id)
            metadata["event_subquery_text_ids"].append(query_id)
            metadata["event_search_indices"].append(search_index)
            metadata["event_raw_step_indices"].append(step_index)
            metadata["event_chunk_content_ids"].append(ids)
            metadata["event_chunk_ranks"].append([1, 2, 3])
            if query:
                metadata["subquery_occurrence_to_event"].append(
                    [subquery_occurrences, event]
                )
                subquery_occurrences += 1
            if previous is not None:
                metadata["event_to_event"].append([previous, event])
            previous = event
    graph_id = f"{dataset}:{qid}"
    sample_ids = [int(record["sample_id"]) for record, _ in kept]
    metadata.update(
        graph_id=graph_id,
        trace_count=len(kept),
        chunk_cross_rollout_coverage=[
            len(coverage[i]) for i in range(len(chunk_texts))
        ],
    )
    scores = [answer_scores(answer, references) for answer in answer_texts]
    return {
        "task": "webqa",
        "dataset": dataset,
        "graph_id": graph_id,
        "question": question,
        "split": question_split(question) if split == "train" else "test",
        "trace_count": len(kept),
        "sample_ids": sample_ids,
        "answer_texts": answer_texts,
        "answer_em": [s[0] for s in scores],
        "answer_f1": [s[1] for s in scores],
        "answer_vote_counts": [votes[i] for i in range(len(answer_texts))],
        "subquery_texts": subquery_texts,
        "chunk_texts": chunk_texts,
        "metadata": metadata,
    }


def message_text(message):
    content = message.get("content") or ""
    if isinstance(content, str):
        return content
    return "\n".join(
        part.get("text") or "" for part in content if isinstance(part, dict)
    )


def browser_events(messages):
    events, pending = [], {}
    for mi, message in enumerate(messages):
        if message.get("role") == "assistant":
            calls = message.get("tool_calls") or []
            if message.get("recipient") in BROWSER_TOOLS:
                calls = [
                    {
                        "id": "native",
                        "function": {
                            "name": message["recipient"],
                            "arguments": message_text(message),
                        },
                    }
                ]
            for call in calls:
                function = call["function"]
                if function["name"] not in BROWSER_TOOLS:
                    continue
                try:
                    arguments = function["arguments"]
                    args = (
                        json.loads(arguments)
                        if isinstance(arguments, str)
                        else arguments
                    )
                    valid = isinstance(args, dict)
                except (ValueError, TypeError):
                    args, valid = {}, False
                event = {
                    "event_id": len(events),
                    "message_index": mi,
                    "tool_type": function["name"],
                    "arguments": args if valid else {},
                    "arguments_valid": valid,
                    "result": "",
                    "result_cursor": None,
                    "result_message_index": None,
                }
                events.append(event)
                pending[call.get("id", "native")] = event
        elif message.get("role") == "tool":
            call_id = message.get("tool_call_id", "native")
            event = pending.get(call_id)
            if (
                event is None
                or message.get("name", event["tool_type"]) != event["tool_type"]
            ):
                continue
            pending.pop(call_id)
            result = message_text(message)
            match = re.match(r"^\s*\[(\d+)\]", result)
            event.update(
                result=result,
                result_message_index=mi,
                result_cursor=int(match[1]) if match else None,
            )
    return track_provenance(events)


def track_provenance(events):
    """Resolve cursors at call time and check any supplied provenance fields."""
    events = sorted(events, key=lambda e: int(e["event_id"]))
    if len({e["event_id"] for e in events}) != len(events):
        raise ValueError("Duplicate browser event ID")
    returns = sorted(
        (e["result_message_index"], e["event_id"], e["result_cursor"])
        for e in events
        if e.get("result_message_index") is not None
        and e.get("result_cursor") is not None
    )
    cursor_counts = Counter(cursor for _, _, cursor in returns)
    result_counts = Counter(mi for mi, _, _ in returns)
    owners, last_cursor, returned = {}, None, 0
    for event in events:
        mi = event["message_index"]
        while returned < len(returns) and returns[returned][0] < mi:
            _, eid, cursor = returns[returned]
            owners.setdefault(cursor, []).append(eid)
            last_cursor = cursor
            returned += 1
        cursor = event["arguments"].get("cursor", -1)
        resolved = last_cursor if cursor == -1 else cursor
        is_search = event["tool_type"] == "browser.search"
        cursor_valid = isinstance(cursor, int) and not isinstance(cursor, bool)
        candidates = owners.get(resolved, []) if cursor_valid and not is_search else []
        parent = (
            candidates[0]
            if len(candidates) == 1
            and event["tool_type"] != "browser.search"
            and event.get("arguments_valid", True)
            else None
        )
        source_cursor = resolved if parent is not None else None
        valid = event.get("arguments_valid", True) and len(candidates) <= 1
        valid &= is_search or cursor_valid
        valid &= (
            event.get("result_message_index") is not None
            and event["result_message_index"] > mi
        )
        valid &= (
            cursor_counts[event.get("result_cursor")] <= 1
            and result_counts[event.get("result_message_index")] <= 1
        )
        for field, value in (
            ("source_event_id", parent),
            ("source_cursor", source_cursor),
        ):
            if field in event and event[field] != value:
                valid = False
            event[field] = value
        event["provenance_valid"] = bool(valid)
    return events


def search_ancestor(event, by_id):
    current, seen = event, set()
    while current["event_id"] not in seen:
        seen.add(current["event_id"])
        if current["tool_type"] == "browser.search":
            return f"search:{current['event_id']}", current["event_id"], False
        parent = by_id.get(current.get("source_event_id"))
        if parent is None:
            return f"orphan:{current['event_id']}", current["event_id"], True
        current = parent
    return f"orphan:{event['event_id']}", event["event_id"], True


def observation_text(event):
    body = event.get("result", "")
    if event["tool_type"] == "browser.find":
        parts = re.split(r"^L\d+: # 【\d+†match at L\d+】[^\n]*\n?", body, flags=re.M)[
            1:
        ]
        return "\n\n".join(
            text
            for part in parts
            if (text := normalize_text(re.sub(r"^L\d+:\s?", "", part, flags=re.M)))
        )
    if not re.search(r"\*\*viewing lines \[\d+ - \d+\] of \d+\*\*", body):
        return ""
    body = re.sub(r"^\s*\[\d+\]\s*", "", body, count=1)
    body = re.sub(r"^\*\*viewing lines.*?\*\*\s*$", "", body, flags=re.M)
    return normalize_text(re.sub(r"^L\d+:\s?", "", body, flags=re.M))


def parse_browser_response(row, corpus=None):
    corpus = corpus or {}
    sample = int(
        row.get("sample_id", row.get("traj_idx", int(row.get("seed", 42)) - 42))
    )
    seed = int(row.get("seed", sample + 42))
    question = row.get(
        "question", row.get("query", corpus.get(row.get("query_content_hash")))
    )
    if "messages" in row:
        messages = row["messages"]
        events = browser_events(messages)
        finals = [
            message_text(m)
            for m in messages
            if m.get("role") == "assistant"
            and not m.get("tool_calls")
            and m.get("recipient") not in BROWSER_TOOLS
            and m.get("channel") == "final"
        ]
        if not finals:
            finals = [
                message_text(m)
                for m in messages
                if m.get("role") == "assistant"
                and m.get("recipient") in (None, "all")
                and not m.get("tool_calls")
                and m.get("channel") is None
            ]
        final = finals[-1] if finals else ""
        if question is None:
            question = next(
                (message_text(m) for m in messages if m.get("role") == "user"), None
            )
    else:
        events = []
        for original in row.get("tool_events", []):
            event = dict(original)
            event.setdefault(
                "arguments_valid", isinstance(event.get("arguments"), dict)
            )
            event["arguments"] = event.get("arguments") or {}
            if "result" not in event:
                parts = event.get("observation_parts") or []
                event["result"] = "\n\n".join(
                    part.get("text", corpus.get(part.get("content_hash"), ""))
                    for part in parts
                )
            events.append(event)
        events = track_provenance(events)
        final = row.get(
            "answer", corpus.get(row.get("candidate_answer_content_hash"), "")
        )
    if not isinstance(question, str) or not question.strip():
        raise ValueError(
            "A browser trajectory needs its question text or query corpus entry"
        )
    answer = row.get("candidate_answer", final)
    if not isinstance(answer, str):
        raise ValueError("Candidate answer must be text")
    correct = row.get("correct")
    if not isinstance(correct, bool):
        raise ValueError("Correctness judgment must be boolean")
    by_id = {e["event_id"]: e for e in events}
    subqueries = {
        f"search:{e['event_id']}": {
            "subquery_id": f"search:{e['event_id']}",
            "event_id": e["event_id"],
            "text": corpus.get(
                e.get("argument_content_hash"),
                normalize_text(e["arguments"].get("query")),
            ),
            "is_orphan": False,
        }
        for e in events
        if e["tool_type"] == "browser.search"
    }
    observations, old_keys = [], {}
    for event in events:
        kind = event["tool_type"]
        if (
            kind not in {"browser.open", "browser.find"}
            or event.get("result_cursor") is None
        ):
            continue
        parts = event.get("observation_parts")
        if parts is not None:
            wanted = "open_observation" if kind == "browser.open" else "find_match"
            selected = [p for p in parts if p.get("kind") == wanted]
            texts = [p.get("text", corpus.get(p.get("content_hash"))) for p in selected]
            if any(t is None for t in texts):
                raise ValueError("Observation text is missing from the supplied corpus")
            text = "\n\n".join(texts)
        else:
            text = observation_text(event)
        if not text:
            continue
        subquery_id, root, orphan = search_ancestor(event, by_id)
        if orphan:
            subqueries.setdefault(
                subquery_id,
                {
                    "subquery_id": subquery_id,
                    "event_id": root,
                    "text": "",
                    "is_orphan": True,
                },
            )
        parent = event.get("source_event_id")
        args = event["arguments"]
        # These local groups are only a fallback for unverified sources.
        if parent in old_keys and (
            kind == "browser.find" or args.get("id", -1) in (-1, None)
        ):
            old_key = old_keys[parent]
        else:
            old_key = (
                canonical_url(args.get("id")) or f"local:{seed}:{event['event_id']}"
            )
        old_keys[event["event_id"]] = old_key
        observations.append(
            {
                "event_id": event["event_id"],
                "tool_kind": 0 if kind == "browser.open" else 1,
                "text": text,
                "doc_key": old_key,
                "subquery_id": subquery_id,
            }
        )
    return question, {
        "seed": seed,
        "sample_id": sample,
        "correct": correct,
        "answer": answer.strip(),
        "subqueries": sorted(subqueries.values(), key=lambda s: s["event_id"]),
        "observations": observations,
        "tool_events": events,
    }


def build_browser_pool(records, dataset, split, k, corpus=None):
    qid = int(records[0]["qid"])
    parsed = [
        parse_browser_response(r, corpus)
        for r in _first_k(
            records,
            k,
            lambda r: int(
                r.get("sample_id", r.get("traj_idx", int(r.get("seed", 42)) - 42))
            ),
        )
    ]
    questions = {q for q, _ in parsed}
    if len(questions) != 1:
        raise ValueError("Conflicting questions within one browsing pool")
    question = questions.pop()
    responses = [response for _, response in parsed if response["answer"]]
    if split == "train" and {r["correct"] for r in responses} != {False, True}:
        return None
    votes = Counter(normalize_answer(r["answer"]) for r in responses)
    documents = {}
    for response in responses:
        response["trajectory_id"] = f"{dataset}:{qid}:{response['seed']}"
        response["vote_count"] = votes[normalize_answer(response["answer"])]
        for observation in response["observations"]:
            key = observation.pop("doc_key")
            documents.setdefault(key, len(documents))
            observation["doc_index"] = documents[key]
    return {
        "task": "long_horizon",
        "qid": qid,
        "dataset": dataset,
        "question": question,
        "split": question_split(question) if split == "train" else "test",
        "responses": responses,
    }
