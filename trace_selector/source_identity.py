"""Resolve document identity from browser returns and navigation provenance."""

import hashlib
import re
import unicodedata
from urllib.parse import quote, unquote, unquote_to_bytes, urlsplit, urlunsplit

RULE_VERSION = "trace-document-identity-v1"
UNRESERVED = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
)


def _display(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def _authority(parsed):
    authority = parsed.netloc.rsplit("@", 1)[-1].lower()
    if parsed.username is None and parsed.password is None:
        return authority
    username = unquote_to_bytes(parsed.username or "")
    material = b"document-userinfo-v1\0" + len(username).to_bytes(8, "big") + username
    material += (
        b"\0" if parsed.password is None else b"\1" + unquote_to_bytes(parsed.password)
    )
    return "userinfo-sha256-" + hashlib.sha256(material).hexdigest() + "@" + authority


def canonical_url(value):
    """Drop fragments without collapsing encoded path/query delimiters."""
    if not isinstance(value, str) or re.search(r"[\x00-\x1f\x7f]", value):
        return None
    try:
        parsed = urlsplit(value)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            return None
        if re.search(r"\s", parsed.netloc) or re.search(
            r"%(?![0-9a-fA-F]{2})", parsed.path + parsed.query
        ):
            return None
        parsed.port  # Validate the port before accepting this identity.

        def component(text, safe):
            encoded = quote(text, safe=safe)

            def escape(match):
                char = chr(int(match[1], 16))
                return char if char in UNRESERVED else "%" + match[1].upper()

            return re.sub(r"%([0-9a-fA-F]{2})", escape, encoded)

        return urlunsplit(
            (
                parsed.scheme.lower(),
                _authority(parsed),
                component(parsed.path, "/:@-._~%"),
                component(parsed.query, "=&?/:@-._~%+"),
                "",
            )
        )
    except (ValueError, UnicodeError):
        return None


def _matching_url_field(tail, displayed):
    tail = re.sub(r"^\s*L\d+:\s?", "", tail, flags=re.M)
    match = re.match(r"\s*URL:\s*", tail)
    if match is None:
        return None
    value = tail[match.end() :]
    candidate = ""
    for i, char in enumerate(value[: max(2048, len(displayed) * 5 + 64)]):
        if char.isspace():
            continue
        candidate += char
        try:
            agrees = _display(unquote(candidate, errors="strict")) == _display(
                displayed
            )
        except UnicodeDecodeError:
            continue
        if agrees and (i + 1 == len(value) or value[i + 1].isspace()):
            if canonical_url(candidate):
                return candidate
    return None


def page_identity(text):
    """Parse the source URL in a returned page header."""
    content = re.sub(r"^\s*\[\d+\]\s*", "", text, count=1)
    raw_lines = "\n**viewing lines " in content or bool(re.search(r"\nL\d+:", content))
    prefix = content.split("\n", 1)[0] if raw_lines else content
    proof = {
        "view_type": "unknown",
        "url": None,
        "displayed_url": None,
        "field_verified": False,
        "status": "missing_returned_header",
    }
    doc = re.match(r"^Doc\s+(\d+)\b", prefix)
    proof["doc_id"] = doc[1] if doc else None
    is_find = prefix.startswith("Find results for text:")
    if re.search(r"\(web-search://[^)]*\)", prefix[:2000]):
        return {**proof, "view_type": "search", "status": "returned_search_view"}
    opening = re.search(r"\(https?://", prefix[:2000])
    if opening is None:
        if is_find:
            proof["view_type"] = "find"
        elif doc:
            proof.update(view_type="document", status="doc_id_without_url_header")
        return proof
    start = opening.start()
    closing = None
    if raw_lines and prefix.rstrip().endswith(")"):
        closing = len(prefix.rstrip()) - 1
    else:
        delimiter = re.search(r"\)\s+URL:\s*", content[start:])
        if delimiter:
            closing = start + delimiter.start()
        else:
            depth = 1
            for i in range(start + 1, min(len(prefix), start + 12000)):
                depth += (prefix[i] == "(") - (prefix[i] == ")")
                if depth == 0:
                    closing = i
                    break
    proof["view_type"] = "find" if is_find else "document"
    if closing is None:
        return {**proof, "status": "ambiguous_header_boundary"}
    displayed = content[start + 1 : closing]
    tail = re.sub(r"^\s*\*\*viewing lines[^\n]*\*\*\s*", "", content[closing + 1 :])
    encoded = _matching_url_field(tail, displayed)
    proof.update(displayed_url=displayed, field_verified=encoded is not None)
    if displayed.rstrip().endswith(("...", "…")) and encoded is None:
        return {**proof, "status": "truncated_returned_url"}
    if is_find:
        marker = displayed.rfind("/find?pattern=")
        proof["url"] = canonical_url(displayed[:marker]) if marker >= 0 else None
        return {**proof, "status": "find_view_requires_parent"}
    proof["url"] = canonical_url(encoded or displayed)
    proof["status"] = (
        "returned_URL_field_matches_display" if encoded else "returned_document_header"
    )
    return proof


def event_identities(events, trajectory_id, doc_support=None):
    """Resolve document sources through returned headers and parent views."""
    states = {}
    doc_support = doc_support or {}
    for event in events:
        eid = event["event_id"]
        kind = event["tool_type"].removeprefix("browser.")
        args = event["arguments"]
        proof = page_identity(event.get("result", "") if kind == "open" else "")
        parent = states.get(event.get("source_event_id"))
        state = {"key": None, "verified": False, "status": "unresolved"}
        valid = (
            event.get("arguments_valid", True)
            and event.get("result_cursor") is not None
            and event.get("provenance_valid", True)
        )
        if not valid:
            state["status"] = "invalid_return_or_provenance"
        elif kind == "search":
            state = {
                "key": f"search-view:{trajectory_id}:{eid}",
                "verified": True,
                "url": None,
                "status": "search_view_identity",
            }
        elif kind == "find" or (kind == "open" and args.get("id", -1) in (-1, None)):
            if parent and parent["verified"]:
                url = proof["url"]
                agrees = url is None or url == parent.get("url")
                displayed = proof.get("displayed_url")
                if not agrees and displayed and not proof["field_verified"]:
                    if proof["view_type"] == "find":
                        displayed = displayed.rsplit("/find?pattern=", 1)[0]
                    parent_url = parent.get("url") or ""
                    parent_display = _display(unquote(parent_url))
                    agrees = (
                        _display(displayed) == parent_display
                        or _display(displayed.split("#", 1)[0]) == parent_display
                    )
                if proof["view_type"] == "search" and parent.get("url"):
                    agrees = False
                if agrees:
                    state = {**parent, "status": f"{kind}_inherits_source"}
                else:
                    state["status"] = "returned_view_conflicts_with_parent"
            # An Open with no parent may still return a complete document header.
            if kind == "open" and not (parent and parent["verified"]):
                if proof["view_type"] == "document":
                    state = _document_state(proof, args, doc_support)
                elif proof["view_type"] == "search":
                    state = {
                        "key": f"search-view:{trajectory_id}:{eid}",
                        "verified": True,
                        "url": None,
                        "status": "returned_search_view_identity",
                    }
        elif kind == "open" and proof["view_type"] == "search":
            state = {
                "key": f"search-view:{trajectory_id}:{eid}",
                "verified": True,
                "url": None,
                "status": "returned_search_view_identity",
            }
        elif kind == "open" and proof["view_type"] == "document":
            state = _document_state(proof, args, doc_support)
        states[eid] = state
    return states


def _document_state(proof, args, doc_support):
    url = proof["url"]
    status = proof["status"]
    if status == "doc_id_without_url_header":
        url = doc_support.get(proof["doc_id"])
        status = "same_question_unique_doc_id_url_support" if url else status
    requested = canonical_url(args.get("id"))
    if (
        url
        and requested
        and not proof["field_verified"]
        and proof["displayed_url"]
        and _display(unquote(requested)) == _display(proof["displayed_url"])
    ):
        url = requested
    return {
        "key": "url:" + url if url else None,
        "url": url,
        "verified": url is not None,
        "status": status,
    }


def build_document_identities(graphs):
    questions = {}
    for graph in graphs:
        qid = graph["qid"]
        support = {}
        for response in graph["responses"]:
            for event in response["tool_events"]:
                proof = page_identity(event.get("result", ""))
                if (
                    event["tool_type"] == "browser.open"
                    and event.get("provenance_valid", True)
                    and event.get("result_cursor") is not None
                    and proof["view_type"] == "document"
                    and proof["doc_id"]
                    and proof["url"]
                ):
                    support.setdefault(proof["doc_id"], set()).add(proof["url"])
        support = {
            doc: next(iter(urls)) for doc, urls in support.items() if len(urls) == 1
        }
        keys, old, memberships, mask, statuses = [], [], [], [], []
        identities = {}
        for response in graph["responses"]:
            states = event_identities(
                response["tool_events"], response["trajectory_id"], support
            )
            for observation in response["observations"]:
                state = states[observation["event_id"]]
                key = (
                    state["key"]
                    if state["verified"]
                    else (f"unresolved_original:{qid}:{observation['doc_index']}")
                )
                identities.setdefault(key, {"key": key, "verified": state["verified"]})
                keys.append([response["seed"], observation["event_id"]])
                old.append(observation["doc_index"])
                memberships.append(key)
                mask.append(state["verified"])
                statuses.append(state["status"])
        ordered = sorted(identities)
        index = {key: i for i, key in enumerate(ordered)}
        questions[str(qid)] = {
            "evidence_keys": keys,
            "old_doc_indices": old,
            "original_doc_indices": sorted(set(old)),
            "identities": [identities[key] for key in ordered],
            "evidence_identity_indices": [index[key] for key in memberships],
            "verified_mask": mask,
            "evidence_status": statuses,
        }
    return {
        "schema_version": "evidence_document_identity_v2",
        "identity_rules_version": RULE_VERSION,
        "questions": questions,
    }
