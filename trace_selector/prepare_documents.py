"""Generate verified document memberships from parsed browser trajectories."""

import argparse
import json
from pathlib import Path

from .prepare import read_records
from .source_identity import build_document_identities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    graphs = list(read_records(args.input))
    if any(graph["task"] != "long_horizon" for graph in graphs):
        raise ValueError("Document identities require parsed browser trajectories")
    if len({graph["qid"] for graph in graphs}) != len(graphs):
        raise ValueError("Duplicate question IDs")
    mapping = build_document_identities(graphs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    masks = [
        verified
        for q in mapping["questions"].values()
        for verified in q["verified_mask"]
    ]
    print(
        {"questions": len(graphs), "observations": len(masks), "verified": sum(masks)}
    )


if __name__ == "__main__":
    main()
