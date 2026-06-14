# Green Wave++ — Evaluation Hardening Plan

My response to `GREENWAVE_FIXES.md` + `adversarial.py`. Short version: the
critique is fair and I agree with almost all of it. The engineering is done;
what's left is making the **evaluation** as honest and complete as the system.
This plan reorders the work by return-on-effort and flags the spots where
reality differs from the dropped-in files.

---

## 0. My take on the critique (where I agree / push back)

| Their point | My call |
|---|---|
| #1 Audio AUC 1.000 looks like an easy test, not a great model | **Agree.** Real gap. Heaviest item though — sequence it after the near-free wins. |
| #2 "Noisy beats perfect" is a trigger-timing artifact, not a finding | **Agree, and I half-said so in the README.** But I won't just flip the wording — I'll *run the controlled experiment* (perfect-knowledge at the same ~80 m trigger) and let the data dictate the honest sentence. |
| #3 Headline hides variance (3× = ±44 s) | **Strongly agree. Highest ROI.** I already have paired seeds in `counterfactual.json`, so CIs + a paired t-test are nearly free. |
| #4 Two unjustified sensor numbers, no sensitivity sweep | **Agree.** The sweep is the most "paper-grade" robustness add. |
| #5 One corridor = "works on my road" | **Agree, lowest urgency.** Real, but it's generalization polish, not a credibility hole. |
| Adversarial experiment missing entirely | **Agree — biggest genuine gap.** I built three trust gates and never measured the thing they exist to prevent. |
| "Wire `RealFusionAdapter` in 5 minutes" | **Push back — see §1.** With my architecture it's ~1 hour, because two of the three gates live *outside* the fusion engine. |

The honesty framing in their §5 is right: every fix makes a number *worse* on
paper and the defense *stronger*. I'm bought in.

---

## 1. Reality check on the provided `adversarial.py`

The script is well-built and the scenarios are exactly the right five. But its
`RealFusionAdapter` guesses an API that isn't mine, and the gates don't all
live where it assumes. Wiring it correctly means:

| Harness expects | My real code | What the adapter must do |
|---|---|---|
| `feed_audio` / `feed_vision` / `tick` (separate) | `fusion.update(audio_conf, audio_bearing, vision_dets, t)` — one call/tick | Buffer dets fed this tick, assemble them, call `update()` once in `tick()` |
| `bearing_err_deg` | engine takes an **absolute** bearing, does the Gaussian internally | `audio_bearing = lane_heading[N] + err` (N→`approach_north`, heading 0) |
| Doppler ×0.3 inside engine | applied **upstream**: `audio_conf *= doppler_factor` in the pipeline | if `det.receding`: multiply confidence by `doppler_recede_factor` before `update()` |
| rate limiter inside engine | `SlidingWindowRateLimiter` in `pipeline._handle_preemption`; fuser fires unconditionally | run returned `FusionCommand`s through a `SlidingWindowRateLimiter`, return lane only if `allow()`; count denials |
| `VisionDet(lane, confidence)` | `_fuse_vision` wants `approaching/distance_m/speed_kmh/speed_mps` | synthesize `approaching=True` + nominal distance/speed so `spoof_flood` can actually fire |
| `belief(lane)` | `fusion.get_beliefs()[lane]` | round-trip through the real getter |

**Upside of doing it right:** the real engine fuses one absolute bearing
across *all four* lanes, so `cross_street` (90° off) is a more faithful test
than the reference model — it'll light up `approach_east`, not the corridor.
That's a better story, not a worse one.

**Estimate:** ~1 focused hour to wire + verify, not 5 minutes. The reference
model still runs out of the box for a sanity baseline; the `--real` number is
the one that goes in the report.

---

## 2. Prioritized plan (ordered by ROI, not calendar)

Effort is **focused work-hours** for me, not student-calendar days.

| # | Task | Kills | Effort | Depends on |
|---|------|-------|--------|-----------|
| ✅ **T1** | **DONE** — `evaluation/significance.py`: paired t-tests, 95% CIs, Cohen's dz, bootstrap, forest plot. Closed-loop significant at every density (p ≤ 3e-6); civilian cost n.s. at 2×/3× | Fix #3 | done | — |
| ✅ **T2** | **DONE** — `evaluation/adversarial.py` wired to the real engine (§1); 0 benign false-fires, spoof bounded; hard CI gate in `tests/test_adversarial.py` | Fix #2's sibling — validates Phases 3–4 | done | — |
| ✅ **T3** | **DONE** — perfect-knowledge re-run at 80 m trigger. Closed-loop matches perfect@80 (gap n.s., p > 0.16); timing effect significant (+10 s @1×, +25 s @2×). Honest reframe: **trigger timing dominates detection accuracy** | Fix #2 | done | — |
| ✅ **T4** | **DONE** — `evaluation/sensitivity.py` sweeps both modalities 20→95% (15 seeds). Time saved is FLAT (62–64s, 100% fire) — temporal accumulation absorbs detector unreliability; cliff is <5% per-tick | Fix #4 — robustness | done | — |
| ✅ **T5** | **DONE — hard audio test set** — sirens mixed into UrbanSound8K at −5..+5 dB SNR, `audio/hard_eval.py` reproduces the clean 1.000 then reports **0.925 overall** (0.86 @ -5 dB, 0.98 @ +5 dB); 1.000 reframed as clean ceiling in the README | Fix #1 — worst credibility liability | done | — |
| ✅ **T6** | **DONE** — built Shollinganallur, Chennai from OSM (`sim/nets/shollinganallur/`); unchanged pipeline runs on it (infrastructure generalizes). Efficacy is corridor-dependent: on this DENSE 10-signal cluster the green wave gave −5.5s (n.s./slightly negative) — honest scope boundary, arterial timings don't suit ~50m spacing. `evaluation/second_corridor.py` | Fix #5 — generalization | done | — |
| ✅ **T7** | **DONE** — `evaluation/latency.py` on the real models (RTX 4060): audio 6.2 ms (16×), vision 7.2 ms (5.6×), fusion 0.003 ms (~30000×) vs real-time budgets. Action delay (~5–6 s) reported separately as deliberate certainty gating, not slowness | Deployability number | done | — |
| ✅ **T8** | **DONE** — `evaluation/route_uncertainty.py`: EV rerouted off-corridor after the first signal (8 seeds). Safety invariant holds — every preemption restores (no stuck greens); cost of a wrong route +0.16s mean civilian delay. Self-heals via the fixed 12s holds | Completeness | done | — |
| ✅ **T9** | **DONE** — `evaluation/adversarial.py --compare`: ours vs naive immediate-preemption. Naive false-fires in 4/4 benign scenarios (731×, incl. 641 from horns); ours 0×. Quantifies what Phases 3–4 buy | Paper requirement | done | — |

**Tiering:**
- **Must-do before any defense:** T1, T2, T3 (≈ half a focused day total, kills the three sharpest questions).
- **Strong robustness adds:** T4, T5.
- **Generalization / paper:** T6–T9.

---

## 3. What I'd do in the very first session (if you say "go")

1. **T1 + T3 together** — one script, `evaluation/significance.py`, reads
   `counterfactual.json`, emits per-density paired t-test + 95% CI, and runs
   the perfect@80 m control. Output: one honest stats table + the corrected
   "perfect vs noisy" sentence. (~2 h, touches no model code.)
2. **T2** — wire `RealFusionAdapter` per §1, run `--real`, drop the verdict
   table into the README's security section, and add it to `tests/` so a
   future gate regression fails CI. (~2 h.)

That's the half-day that turns "strong system, holey eval" into
"defensible." Everything else is incremental on top.

---

## 4. What I would NOT do yet

Agreeing with their §3 closing note:

- **No live hardware capture** (mic/camera threads) — it's the one item left
  on the original 5-phase plan, but a feature bolted onto a holey eval is
  weaker, not stronger.
- **No multi-ambulance arbitration** — Future Work, same reason.
- **Won't silently drop the "beats perfect" line** — I'll replace it with
  whatever T3's controlled run actually shows, stated plainly.

---

## 5. Decisions I need from you

1. **Second corridor (T6):** which city/junction? The critique suggested
   Nellore or Chennai. Pick one with a clearly different topology from Benz
   Circle (e.g. a longer 5-signal arterial) and I'll script the OSM pull.
2. **Adversarial as a CI gate (T2):** want it wired as a hard test-suite
   failure (build breaks if a benign scenario ever fires), or just a
   reportable script? I'd recommend the hard gate.
3. **Scope for now:** do you want me to execute **T1–T3** (the must-do
   half-day) next, or just keep this as a plan for you to schedule?

---

*Nothing in this plan has been executed — it's the map. The dropped-in
`adversarial.py` is still in `Downloads/`; T2 moves it into `evaluation/` and
wires it to the real engine.*
