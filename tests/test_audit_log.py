"""Audit log: the hash chain holds, and every kind of tampering gets caught."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.audit import GENESIS, AuditLog, verify  # noqa: E402


def _fake_clock(start=1000.0, step=0.5):
    t = [start]
    def now():
        t[0] += step
        return t[0]
    return now


def test_chain_builds_and_verifies(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl", now=_fake_clock())
    log.append("pipeline_start", {"demo": True})
    log.append("preempt_fired", {"lane": "approach_north", "belief": 0.91})
    log.append("preempt_denied_rate_limit", {"lane": "approach_north"})

    ok, bad = verify(tmp_path / "audit.jsonl")
    assert ok and bad is None

    lines = [json.loads(l) for l in (tmp_path / "audit.jsonl").read_text().splitlines()]
    assert [e["seq"] for e in lines] == [0, 1, 2]
    assert lines[0]["prev_hash"] == GENESIS
    assert lines[1]["prev_hash"] == lines[0]["hash"]
    assert lines[2]["prev_hash"] == lines[1]["hash"]


def test_edited_payload_is_detected(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path, now=_fake_clock())
    for i in range(4):
        log.append("preempt_fired", {"lane": "approach_west", "n": i})

    # Quietly claim entry 2 happened on a different lane
    lines = path.read_text().splitlines()
    doctored = json.loads(lines[2])
    doctored["payload"]["lane"] = "approach_east"
    lines[2] = json.dumps(doctored, sort_keys=True)
    path.write_text("\n".join(lines) + "\n")

    ok, bad = verify(path)
    assert not ok and bad == 2


def test_deleted_line_is_detected(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path, now=_fake_clock())
    for i in range(3):
        log.append("event", {"n": i})

    lines = path.read_text().splitlines()
    del lines[1]
    path.write_text("\n".join(lines) + "\n")

    ok, bad = verify(path)
    assert not ok and bad == 1


def test_reordered_lines_are_detected(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path, now=_fake_clock())
    for i in range(3):
        log.append("event", {"n": i})

    lines = path.read_text().splitlines()
    lines[1], lines[2] = lines[2], lines[1]
    path.write_text("\n".join(lines) + "\n")

    ok, bad = verify(path)
    assert not ok and bad == 1


def test_reopen_resumes_chain(tmp_path):
    path = tmp_path / "audit.jsonl"
    AuditLog(path, now=_fake_clock()).append("pipeline_start", {})

    # New process, same file: the chain must continue, not restart
    log2 = AuditLog(path, now=_fake_clock(2000.0))
    log2.append("pipeline_stop", {})

    ok, _ = verify(path)
    assert ok
    entries = [json.loads(l) for l in path.read_text().splitlines()]
    assert [e["seq"] for e in entries] == [0, 1]
    assert entries[1]["prev_hash"] == entries[0]["hash"]


def test_missing_file_is_trivially_intact(tmp_path):
    ok, bad = verify(tmp_path / "never_written.jsonl")
    assert ok and bad is None
