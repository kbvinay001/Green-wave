#!/usr/bin/env python3
"""
Hash-chained audit log -- Green Wave++

Every security-relevant event (preemption fired, preemption denied, fusion
reset, pipeline start/stop) lands here as one JSONL line chained to the
previous one by SHA-256. Flip a single byte anywhere in the file -- or
delete or reorder a line -- and verify() points at the first broken entry.
Nobody can quietly rewrite history about when the lights went green.

Chain rule:
    hash_i = sha256(prev_hash_i + canonical_json(core_i))

where core_i is the entry minus its own hash, canonical_json sorts keys
with no whitespace (so dict ordering can't change the hash), and the first
entry chains to 64 zeros.

Verify a log from the shell:
    python -m common.audit logs/audit.jsonl
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Optional, Tuple

GENESIS = "0" * 64

_CORE_KEYS = ("seq", "ts", "event", "payload", "prev_hash")


def _canonical(core: dict) -> str:
    return json.dumps(core, sort_keys=True, separators=(",", ":"))


def _chain_hash(prev_hash: str, core: dict) -> str:
    return hashlib.sha256((prev_hash + _canonical(core)).encode()).hexdigest()


class AuditLog:
    """Append-only writer. Re-opening an existing file resumes its chain."""

    def __init__(self, path, now=time.time):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._now = now
        self._seq, self._prev = self._load_tail()

    def _load_tail(self) -> Tuple[int, str]:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return 0, GENESIS
        last = None
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    last = line
        entry = json.loads(last)
        return int(entry["seq"]) + 1, entry["hash"]

    def append(self, event: str, payload: Optional[dict] = None) -> dict:
        core = {
            "seq":       self._seq,
            "ts":        round(self._now(), 3),
            "event":     event,
            "payload":   payload or {},
            "prev_hash": self._prev,
        }
        entry = dict(core, hash=_chain_hash(self._prev, core))
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")
        self._seq  += 1
        self._prev  = entry["hash"]
        return entry


def verify(path) -> Tuple[bool, Optional[int]]:
    """
    Walk the whole chain. Returns (True, None) when intact, otherwise
    (False, seq) where seq is the first entry that fails -- whether its
    content was edited, its predecessor was removed, or lines were swapped.
    A missing or empty file is trivially intact.
    """
    p = Path(path)
    if not p.exists():
        return True, None

    prev, expected_seq = GENESIS, 0
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                return False, expected_seq
            core = {k: entry[k] for k in _CORE_KEYS if k in entry}
            if (entry.get("seq")       != expected_seq
                    or entry.get("prev_hash") != prev
                    or entry.get("hash")      != _chain_hash(prev, core)):
                return False, expected_seq
            prev          = entry["hash"]
            expected_seq += 1
    return True, None


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Verify an audit log's hash chain")
    ap.add_argument("logfile", nargs="?", default="logs/audit.jsonl")
    args = ap.parse_args()

    ok, bad_seq = verify(args.logfile)
    if ok:
        print(f"OK -- chain intact: {args.logfile}")
    else:
        print(f"TAMPERED -- chain breaks at seq {bad_seq}: {args.logfile}")
        raise SystemExit(1)
