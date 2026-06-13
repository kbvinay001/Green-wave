"""
evaluation/adversarial.py
==========================
Adversarial / false-preemption experiment for Green Wave++.

WHY THIS EXISTS
---------------
The counterfactual eval proves the system HELPS when a real ambulance is
present. It does NOT prove the system stays QUIET when one is absent. The
three trust gates (cross-modal cap, Doppler suppression, rate limit) exist
specifically to prevent false green waves -- but nothing measured whether
they hold end to end. This is that measurement.

THE HEADLINE NUMBER
-------------------
False preemptions across all benign adversarial scenarios must be ZERO.
The only scenario allowed to fire is a PERFECT dual-modal spoof, and even
then it is bounded -- by the fusion state machine (one fire, then ACTIVE)
and the rate limiter (<= max_preempts_per_lane per window) behind it. The
script exits non-zero on any unexpected preemption, so it doubles as a CI
regression gate (see tests/test_adversarial.py).

THE ENGINE UNDER TEST
---------------------
By default this drives the REAL `fusion.fuser.TemporalFusionEngine` through
`RealFusionAdapter`, with the production `common/config.yaml`. That adapter
reproduces exactly what the live pipeline does around the engine:

  - Doppler weighting is applied UPSTREAM (audio_conf *= 0.3 when receding),
    because the real engine never sees a "receding" flag -- the stream
    detector folds it into the confidence before fusion.
  - The rate limiter is a SEPARATE layer (common.rate_limiter), exactly as
    in pipeline._handle_preemption -- the fuser fires commands
    unconditionally and the limiter gates them.
  - A single absolute audio bearing is fused across ALL lanes, so a
    cross-street siren correctly lights up the perpendicular approach, not
    the corridor.

`--reference` swaps in a self-contained re-implementation of the gates
(ReferenceFusionAdapter) for a sanity baseline that needs no engine import.

HOW TO RUN
----------
    python -m evaluation.adversarial              # real engine (the report number)
    python -m evaluation.adversarial --seeds 5    # quick look
    python -m evaluation.adversarial --reference  # toy gate model, tests itself
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# --------------------------------------------------------------------------
# Config: pulled from common/config.yaml when present, else README defaults.
# --------------------------------------------------------------------------
DEFAULTS = {
    "arm_threshold": 0.6,
    "arm_duration_sec": 0.5,
    "preempt_threshold": 0.8,
    "decay_factor": 0.92,            # per second
    "sigma_angle_deg": 20.0,
    "audio_cap": 0.7,                # cross-modal: audio alone can't exceed this
    "doppler_recede_factor": 0.3,    # receding sirens weighted x0.3
    "vision_confirm_window_sec": 3.0,
    "max_preempts_per_lane": 4,      # security.rate_limit
    "rate_window_sec": 3600.0,
}


def load_config() -> dict:
    cfg = dict(DEFAULTS)
    p = ROOT / "common" / "config.yaml"
    if p.exists():
        try:
            import yaml  # type: ignore
            raw = yaml.safe_load(p.read_text()) or {}
            fusion = raw.get("fusion", {})
            sec = raw.get("security", {})
            rl = sec.get("rate_limit", {})
            dop = fusion.get("doppler", {})
            cfg.update({
                "arm_threshold": fusion.get("arm_threshold", cfg["arm_threshold"]),
                "arm_duration_sec": fusion.get("arm_duration_sec", cfg["arm_duration_sec"]),
                "preempt_threshold": fusion.get("preempt_threshold", cfg["preempt_threshold"]),
                "decay_factor": fusion.get("decay_factor", cfg["decay_factor"]),
                "sigma_angle_deg": fusion.get("sigma_angle_deg", cfg["sigma_angle_deg"]),
                "audio_cap": fusion.get("audio_only_belief_cap", cfg["audio_cap"]),
                "doppler_recede_factor": dop.get("receding_factor", cfg["doppler_recede_factor"]),
                "vision_confirm_window_sec": fusion.get("vision_confirm_window_sec",
                                                        cfg["vision_confirm_window_sec"]),
                "max_preempts_per_lane": rl.get("max_preempts_per_lane", cfg["max_preempts_per_lane"]),
                "rate_window_sec": rl.get("window_sec", cfg["rate_window_sec"]),
            })
            print(f"[config] loaded overrides from {p}")
        except Exception as e:  # noqa: BLE001
            print(f"[config] could not parse {p} ({e}); using defaults")
    return cfg


# --------------------------------------------------------------------------
# Detection events fed into the fusion logic each tick.
# --------------------------------------------------------------------------
@dataclass
class AudioDet:
    lane: str
    bearing_err_deg: float   # |measured bearing - lane heading|
    confidence: float
    receding: bool


@dataclass
class VisionDet:
    lane: str
    confidence: float


DT = 0.1  # 10 Hz, matches the fusion loop


# ==========================================================================
# ADAPTERS
# ==========================================================================
class FusionAdapter:
    """Interface the harness expects."""

    def feed_audio(self, det: AudioDet, t: float) -> None: ...
    def feed_vision(self, det: VisionDet, t: float) -> None: ...
    def tick(self, t: float) -> Optional[str]:
        """Advance one DT. Return the lane name if a preemption FIRED and was
        allowed through the rate limiter this tick, else None."""
        ...
    def belief(self, lane: str) -> float: ...
    def denied_count(self) -> int: ...


class _RateLimiter:
    def __init__(self, max_per_lane: int, window: float):
        self.max = max_per_lane
        self.window = window
        self.hits: dict[str, deque] = {}

    def allow(self, lane: str, t: float) -> bool:
        q = self.hits.setdefault(lane, deque())
        while q and t - q[0] > self.window:
            q.popleft()
        if len(q) >= self.max:
            return False
        q.append(t)
        return True


class ReferenceFusionAdapter(FusionAdapter):
    """
    Faithful re-implementation of the gates described in the README. Runs with
    no engine import, but it tests ITSELF, not the real code -- a sanity
    baseline only. The default is RealFusionAdapter.
    """

    def __init__(self, cfg: dict, lanes: list[str]):
        self.cfg = cfg
        self.lanes = lanes
        self.b = {ln: 0.0 for ln in lanes}
        self.last_vision = {ln: -1e9 for ln in lanes}
        self.armed_ticks = {ln: 0 for ln in lanes}
        self.rl = _RateLimiter(cfg["max_preempts_per_lane"], cfg["rate_window_sec"])
        self._denied = 0
        self._pending_audio: list[AudioDet] = []
        self._pending_vision: list[VisionDet] = []

    def feed_audio(self, det: AudioDet, t: float) -> None:
        self._pending_audio.append(det)

    def feed_vision(self, det: VisionDet, t: float) -> None:
        self._pending_vision.append(det)

    def _gauss(self, err_deg: float) -> float:
        s = self.cfg["sigma_angle_deg"]
        return math.exp(-(err_deg ** 2) / (2.0 * s * s))

    def tick(self, t: float) -> Optional[str]:
        cfg = self.cfg
        decay = cfg["decay_factor"] ** DT
        for ln in self.lanes:
            self.b[ln] *= decay
        for d in self._pending_audio:
            contrib = d.confidence * self._gauss(d.bearing_err_deg)
            if d.receding:
                contrib *= cfg["doppler_recede_factor"]
            self.b[d.lane] += contrib * (1.0 - self.b[d.lane])
        for d in self._pending_vision:
            self.last_vision[d.lane] = t
            self.b[d.lane] += d.confidence * (1.0 - self.b[d.lane])
        self._pending_audio.clear()
        self._pending_vision.clear()
        fired = None
        for ln in self.lanes:
            recent_vision = (t - self.last_vision[ln]) <= cfg["vision_confirm_window_sec"]
            if not recent_vision:
                self.b[ln] = min(self.b[ln], cfg["audio_cap"])
            self.b[ln] = max(0.0, min(1.0, self.b[ln]))
            if self.b[ln] >= cfg["arm_threshold"]:
                self.armed_ticks[ln] += 1
            else:
                self.armed_ticks[ln] = 0
            held = self.armed_ticks[ln] * DT >= cfg["arm_duration_sec"]
            if self.b[ln] >= cfg["preempt_threshold"] and recent_vision and held:
                if self.rl.allow(ln, t):
                    fired = ln
                else:
                    self._denied += 1
        return fired

    def belief(self, lane: str) -> float:
        return self.b[lane]

    def denied_count(self) -> int:
        return self._denied


class RealFusionAdapter(FusionAdapter):
    """
    Drives the actual fusion.fuser.TemporalFusionEngine with the production
    config, reproducing the pipeline's behaviour around it (see module
    docstring). This is the adapter whose result goes in the report.
    """

    # harness lane -> (real engine lane name, lane heading in degrees)
    LANE_MAP = {"N": ("approach_north", 0.0), "E": ("approach_east", 90.0),
                "S": ("approach_south", 180.0), "W": ("approach_west", 270.0)}
    # vision stand-ins: a confident, clearly-approaching detection so the
    # cross-modal gate can lift -- the spoof has a real camera track.
    VISION_DISTANCE_M = 60.0
    VISION_SPEED_KMH  = 50.0

    def __init__(self, cfg: dict, lanes: list[str]):
        import yaml
        from fusion.fuser import Lane, TemporalFusionEngine
        from common.rate_limiter import SlidingWindowRateLimiter

        real_cfg = yaml.safe_load((ROOT / "common" / "config.yaml").read_text())
        real_lanes = [
            Lane(self.LANE_MAP[ln][0], heading_deg=self.LANE_MAP[ln][1],
                 corridor_tls=[f"{ln}_tls1", f"{ln}_tls2"])
            for ln in lanes
        ]
        self.engine = TemporalFusionEngine(real_lanes, real_cfg)
        self.rl = SlidingWindowRateLimiter(
            max_events=int(cfg["max_preempts_per_lane"]),
            window_sec=float(cfg["rate_window_sec"]),
        )
        self.doppler_factor = float(cfg["doppler_recede_factor"])
        self.name_to_real = {ln: self.LANE_MAP[ln][0] for ln in lanes}
        self.real_to_name = {real: ln for ln, (real, _) in self.LANE_MAP.items()}
        self._denied = 0
        self._buf_audio: Optional[AudioDet] = None
        self._buf_vision: list[VisionDet] = []

    def feed_audio(self, det: AudioDet, t: float) -> None:
        self._buf_audio = det          # latest wins, mirrors the queue drain

    def feed_vision(self, det: VisionDet, t: float) -> None:
        self._buf_vision.append(det)

    def tick(self, t: float) -> Optional[str]:
        audio_conf, bearing = 0.0, None
        if self._buf_audio is not None:
            d = self._buf_audio
            heading = self.LANE_MAP[d.lane][1]
            bearing = heading + d.bearing_err_deg        # absolute bearing
            audio_conf = d.confidence * (self.doppler_factor if d.receding else 1.0)

        vision_dets = [{
            "lane_id":    self.name_to_real[v.lane],
            "confidence": v.confidence,
            "approaching": True,
            "distance_m": self.VISION_DISTANCE_M,
            "speed_kmh":  self.VISION_SPEED_KMH,
            "speed_mps":  self.VISION_SPEED_KMH / 3.6,
        } for v in self._buf_vision]

        self._buf_audio = None
        self._buf_vision = []

        commands = self.engine.update(audio_conf, bearing, vision_dets, t)

        fired = None
        for cmd in commands:               # rate limiter gates, as in the pipeline
            if self.rl.allow(cmd.target_lane, t):
                fired = self.real_to_name.get(cmd.target_lane, cmd.target_lane)
            else:
                self._denied += 1
        return fired

    def belief(self, lane: str) -> float:
        return self.engine.get_beliefs()[self.name_to_real[lane]]

    def denied_count(self) -> int:
        return self._denied


class NaiveFusionAdapter(FusionAdapter):
    """
    Prior-art strawman: immediate preemption the instant any single detection
    crosses a fire threshold -- no cross-modal gate, no Doppler, no bearing
    kernel, no arm-hold, no rate limit. This is the classic "trigger the green
    on siren detected" acoustic/optical EVP (Opticom-style). It exists here to
    quantify exactly what the trust gates buy: run it through the same
    scenarios and watch it fire on things the gated engine ignores.
    """

    FIRE_THR = 0.5     # a typical raw detection threshold

    def __init__(self, cfg: dict, lanes: list[str]):
        self.lanes = lanes
        self._buf_audio: Optional[AudioDet] = None
        self._buf_vision: list[VisionDet] = []
        self._firing = False         # latch so we count episodes, not ticks
        self._belief = 0.0

    def feed_audio(self, det: AudioDet, t: float) -> None:
        self._buf_audio = det

    def feed_vision(self, det: VisionDet, t: float) -> None:
        self._buf_vision.append(det)

    def tick(self, t: float) -> Optional[str]:
        a_conf = self._buf_audio.confidence if self._buf_audio else 0.0
        v_conf = max((d.confidence for d in self._buf_vision), default=0.0)
        lane = (self._buf_audio.lane if self._buf_audio
                else self._buf_vision[0].lane if self._buf_vision else None)
        self._belief = max(a_conf, v_conf)
        self._buf_audio, self._buf_vision = None, []

        fired = None
        if self._belief >= self.FIRE_THR:
            if not self._firing:      # rising edge = a new preemption episode
                fired = lane
            self._firing = True
        else:
            self._firing = False
        return fired

    def belief(self, lane: str) -> float:
        return self._belief

    def denied_count(self) -> int:
        return 0


# ==========================================================================
# SCENARIOS  (corridor lane under test = "N")
# ==========================================================================
CORRIDOR_LANE = "N"
LANES = ["N", "E", "S", "W"]
DURATION_S = 30.0


def gen_phone_speaker(rng):
    """Loud siren from a phone, on-corridor bearing, but NO camera ever.
    Tests the cross-modal gate: audio alone must cap at 0.7 and never fire."""
    for k in range(int(DURATION_S / DT)):
        t = k * DT
        a = v = None
        if 2.0 <= t <= 27.0:
            a = AudioDet(CORRIDOR_LANE, bearing_err_deg=rng.gauss(0, 4),
                         confidence=min(0.99, max(0.0, rng.gauss(0.88, 0.05))),
                         receding=False)
        yield t, a, v


def gen_receding_ev(rng):
    """A real ambulance that has ALREADY PASSED: falling pitch (Doppler),
    no fresh inbound camera lock. Only lingering evidence is receding audio.
    Tests the Doppler gate (x0.3) + the cross-modal cap together."""
    for k in range(int(DURATION_S / DT)):
        t = k * DT
        a = v = None
        if 2.0 <= t <= 22.0:
            a = AudioDet(CORRIDOR_LANE, bearing_err_deg=rng.gauss(20, 8),
                         confidence=min(0.99, max(0.0, rng.gauss(0.82, 0.05))),
                         receding=True)
        yield t, a, v


def gen_cross_street(rng):
    """Siren on a perpendicular road: ~90 deg off the corridor bearing, no
    corridor vision. Tests the bearing kernel + cross-modal gate."""
    for k in range(int(DURATION_S / DT)):
        t = k * DT
        a = v = None
        if 2.0 <= t <= 27.0:
            a = AudioDet(CORRIDOR_LANE, bearing_err_deg=rng.gauss(90, 8),
                         confidence=min(0.99, max(0.0, rng.gauss(0.88, 0.05))),
                         receding=False)
        yield t, a, v


def gen_noise_burst(rng):
    """Intermittent CRNN false alarms (car horns, etc.), never sustained, no
    vision. Tests decay + arm-hold: must never even arm long enough."""
    for k in range(int(DURATION_S / DT)):
        t = k * DT
        a = v = None
        if rng.random() < 0.08 and t > 1.0:
            a = AudioDet(CORRIDOR_LANE, bearing_err_deg=rng.gauss(0, 10),
                         confidence=max(0.0, rng.gauss(0.65, 0.1)),
                         receding=False)
        yield t, a, v


def gen_spoof_flood(rng):
    """Worst case: attacker fakes BOTH audio AND vision continuously. The
    system CANNOT distinguish a perfect dual-modal spoof from a real EV -- so
    it WILL fire. The defence is twofold: the fusion state machine fires once
    then latches ACTIVE, and the rate limiter bounds re-fires to
    max_preempts_per_lane. Allowed to fire, but only bounded."""
    for k in range(int(DURATION_S / DT)):
        t = k * DT
        a = v = None
        if t > 1.0:
            a = AudioDet(CORRIDOR_LANE, bearing_err_deg=rng.gauss(0, 3),
                         confidence=min(0.99, max(0.0, rng.gauss(0.92, 0.03))),
                         receding=False)
            v = VisionDet(CORRIDOR_LANE, confidence=min(0.99, max(0.0, rng.gauss(0.90, 0.03))))
        yield t, a, v


SCENARIOS = {
    "phone_speaker":  (gen_phone_speaker,  "cross_modal_cap",    0),
    "receding_ev":    (gen_receding_ev,    "doppler+decay",      0),
    "cross_street":   (gen_cross_street,   "bearing_kernel",     0),
    "noise_burst":    (gen_noise_burst,    "arm_hold+decay",     0),
    "spoof_flood":    (gen_spoof_flood,    "rate_limit",        -1),  # -1 = bounded
}


@dataclass
class ScenarioResult:
    name: str
    expected: int               # 0 = must never fire; -1 = bounded by rate limit
    guard: str
    seeds: int
    fires_per_seed: list[int] = field(default_factory=list)
    max_belief_per_seed: list[float] = field(default_factory=list)
    denied_per_seed: list[int] = field(default_factory=list)

    @property
    def total_fires(self) -> int:
        return sum(self.fires_per_seed)

    @property
    def max_fires_in_any_seed(self) -> int:
        return max(self.fires_per_seed) if self.fires_per_seed else 0

    @property
    def peak_belief(self) -> float:
        return max(self.max_belief_per_seed) if self.max_belief_per_seed else 0.0


def _adapter_cls(kind: str):
    return {"real": RealFusionAdapter,
            "reference": ReferenceFusionAdapter,
            "naive": NaiveFusionAdapter}[kind]


def run_scenario(name, gen_fn, guard, expected, cfg, seeds, use_real,
                 adapter: str | None = None) -> ScenarioResult:
    res = ScenarioResult(name=name, expected=expected, guard=guard, seeds=seeds)
    AdapterCls = _adapter_cls(adapter or ("real" if use_real else "reference"))
    for s in range(seeds):
        rng = random.Random(1000 + s)
        eng = AdapterCls(cfg, LANES)
        fires = 0
        peak = 0.0
        for t, a, v in gen_fn(rng):
            if a is not None:
                eng.feed_audio(a, t)
            if v is not None:
                eng.feed_vision(v, t)
            fired = eng.tick(t)
            peak = max(peak, eng.belief(CORRIDOR_LANE))
            if fired is not None:
                fires += 1
        res.fires_per_seed.append(fires)
        res.max_belief_per_seed.append(round(peak, 3))
        res.denied_per_seed.append(eng.denied_count())
    return res


def run_all(cfg: dict, seeds: int, use_real: bool,
            adapter: str | None = None) -> list[ScenarioResult]:
    return [run_scenario(name, gen_fn, guard, expected, cfg, seeds, use_real, adapter)
            for name, (gen_fn, guard, expected) in SCENARIOS.items()]


def run_comparison(cfg: dict, seeds: int) -> dict:
    """
    Ours (gated, real engine) vs a naive immediate-preemption baseline, on the
    same scenarios. The prior-art comparison: it quantifies what the trust
    gates buy by counting false preemptions a naive system would suffer.
    """
    gated = {r.name: r for r in run_all(cfg, seeds, use_real=True, adapter="real")}
    naive = {r.name: r for r in run_all(cfg, seeds, use_real=False, adapter="naive")}
    rows = []
    for name, (_, guard, expected) in SCENARIOS.items():
        rows.append({
            "scenario": name, "guard": guard, "expected": expected,
            "naive_fires": naive[name].total_fires,
            "gated_fires": gated[name].total_fires,
        })
    benign = [r for r in rows if r["expected"] == 0]
    return {
        "seeds": seeds,
        "rows": rows,
        "naive_benign_false_fires": sum(r["naive_fires"] for r in benign),
        "gated_benign_false_fires": sum(r["gated_fires"] for r in benign),
        "benign_scenarios": len(benign),
        "naive_scenarios_leaking": sum(1 for r in benign if r["naive_fires"] > 0),
    }


def evaluate(results: list[ScenarioResult], cfg: dict) -> tuple[bool, list[str]]:
    """Return (all_pass, failure_messages)."""
    ok = True
    msgs = []
    cap = cfg["max_preempts_per_lane"]
    for r in results:
        if r.expected == 0:
            if r.total_fires != 0:
                ok = False
                msgs.append(f"FAIL [{r.name}]: {r.total_fires} false preemption(s) "
                            f"across {r.seeds} seeds (must be 0). Guard '{r.guard}' leaked.")
        elif r.expected == -1:
            if r.max_fires_in_any_seed > cap:
                ok = False
                msgs.append(f"FAIL [{r.name}]: {r.max_fires_in_any_seed} fires in a seed "
                            f"exceeds rate cap {cap}. Rate limiter not bounding spoofs.")
    return ok, msgs


def print_table(results: list[ScenarioResult], cfg: dict):
    print("\n" + "=" * 78)
    print(f"{'scenario':<16}{'guard':<18}{'peak belief':<13}{'fires (max/seed)':<18}{'verdict'}")
    print("-" * 78)
    cap = cfg["max_preempts_per_lane"]
    for r in results:
        if r.expected == 0:
            verdict = "PASS" if r.total_fires == 0 else f"LEAK x{r.total_fires}"
            fires_str = f"{r.total_fires} total"
        else:
            verdict = "PASS" if r.max_fires_in_any_seed <= cap else "UNBOUNDED"
            fires_str = f"{r.max_fires_in_any_seed} (cap {cap})"
        print(f"{r.name:<16}{r.guard:<18}{r.peak_belief:<13.3f}{fires_str:<18}{verdict}")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser(description="Green Wave++ adversarial false-preemption experiment")
    ap.add_argument("--seeds", type=int, default=30, help="seeds per scenario (default 30)")
    ap.add_argument("--reference", action="store_true",
                    help="use the self-contained reference gate model instead of the real engine")
    ap.add_argument("--compare", action="store_true",
                    help="ours (gated) vs a naive immediate-preemption baseline -- prior-art comparison")
    ap.add_argument("--out", default="evaluation/results/adversarial.json")
    args = ap.parse_args()

    cfg = load_config()

    if args.compare:
        comp = run_comparison(cfg, args.seeds)
        print(f"\n[compare] gated (ours) vs naive immediate preemption  ({args.seeds} seeds/scenario)")
        print("=" * 70)
        print(f"{'scenario':<16}{'guard':<18}{'naive fires':<14}{'ours (gated)'}")
        print("-" * 70)
        for r in comp["rows"]:
            tag = "" if r["expected"] == 0 else "  (spoof: both fire)"
            print(f"{r['scenario']:<16}{r['guard']:<18}{r['naive_fires']:<14}{r['gated_fires']}{tag}")
        print("=" * 70)
        print(f"benign scenarios: naive false-fires in "
              f"{comp['naive_scenarios_leaking']}/{comp['benign_scenarios']} "
              f"({comp['naive_benign_false_fires']} total) | ours: "
              f"{comp['gated_benign_false_fires']}")
        out = ROOT / "evaluation" / "results" / "adversarial_compare.json"
        out.write_text(json.dumps(comp, indent=2))
        print(f"[out] wrote {out}")
        return
    use_real = not args.reference
    engine_name = "REAL TemporalFusionEngine" if use_real else "REFERENCE model (re-implementation)"
    print(f"[engine] {engine_name}")
    if not use_real:
        print("[engine] NOTE: the reference model tests itself, not the real code. "
              "Drop --reference for the report number.")

    results = run_all(cfg, args.seeds, use_real)

    print_table(results, cfg)
    all_pass, msgs = evaluate(results, cfg)
    for m in msgs:
        print(m)

    out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "engine": "real" if use_real else "reference",
        "seeds_per_scenario": args.seeds,
        "config": cfg,
        "all_pass": all_pass,
        "scenarios": [
            {
                "name": r.name, "guard": r.guard, "expected": r.expected,
                "total_fires": r.total_fires, "max_fires_in_any_seed": r.max_fires_in_any_seed,
                "peak_belief": r.peak_belief,
                "fires_per_seed": r.fires_per_seed,
                "denied_per_seed": r.denied_per_seed,
            } for r in results
        ],
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\n[out] wrote {out}")

    if all_pass:
        print("\nRESULT: PASS -- 0 false preemptions in benign scenarios; spoof bounded.")
    else:
        print("\nRESULT: FAIL -- a gate is leaking; see messages above.")
    raise SystemExit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
