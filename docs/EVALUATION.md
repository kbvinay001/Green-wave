# How we know it works

This is the long version of the evaluation — every claim in the README's
summary table, with the method, the numbers, and how to reproduce each one.
Nothing here is asserted; it all comes out of a script you can run.

All the SUMO numbers come straight from `tripinfo` output. No assumed cycle
times, no analytical baselines, no cherry-picked seeds.

---

## Does it actually help a real ambulance?

The question that decides whether any of this matters: with the **same
traffic, same routes, same random seed**, how much faster does the ambulance
cross Benz Circle when the green wave fires — and what does it cost everyone
else? `evaluation/counterfactual.py` runs paired headless SUMO simulations on
the real network across **three demand levels × 20 random seeds × three
modes** (180 runs).

![The real corridor](img/benz_circle_network.png)

The three modes are the heart of it:

- **baseline** — signals run their normal programs; the ambulance queues like
  any other vehicle. The control.
- **green wave (perfect)** — the cascade arms from the ambulance's
  ground-truth position the instant it's within 220 m. The theoretical
  ceiling: what flawless detection would buy.
- **closed loop (noisy sensors)** — the one that matters. The ambulance's
  position becomes *imperfect evidence* (synthetic audio within 150 m at an
  85% hit rate with noisy bearing; synthetic vision within 80 m at 90% with
  noisy confidence) and feeds the **unchanged `TemporalFusionEngine`** —
  every gate, the arm hold, the cross-modal cap — which decides when the
  preemption actually fires. The real pipeline driving real signals; only the
  sensor front-end is faked.

The eval ambulance deliberately does **not** carry SUMO's `bluelight` device
— that model parts traffic into a perfect rescue lane and runs red lights,
which is exactly the citizen-cooperation assumption that fails on the
arterials this targets. (With bluelight on, the measured benefit is noise,
−6 s to +3 s.) The eval ambulance obeys signals and queues like everything
else, so the measurement isolates exactly what the signal preemption removes.

Headline (mean ± std over 20 seeds, corridor travel time):

| demand | baseline | closed loop | time saved | stops | cost per civilian |
|---|---|---|---|---|---|
| 1× recorded | 176.5 s | 114.0 s | **62.5 ± 6.5 s (35.4%)** | 4.1 → 1.3 | +0.8 s |
| 2× | 181.1 s | 129.5 s | **51.6 ± 16.2 s (28.5%)** | 4.2 → 1.8 | +0.3 s |
| 3× | 197.6 s | 133.3 s | **64.4 ± 44.1 s (32.6%)** | 4.8 → 1.9 | −0.6 s |

![Counterfactual results](img/counterfactual_travel_time.png)

The ambulance crosses **~29–35% faster at every density**. The right-hand
panel above is the honesty metric: all three civilian-cost curves sit on top
of each other. The green wave buys the ambulance a minute for under a second
of added delay per civilian — and at 3× demand the corridor clears so cleanly
that traffic behind the ambulance actually comes out slightly ahead.

The speed trace is the whole argument in one picture — all three runs are
identical until each arms its cascade, then the baseline ambulance dead-stops
in signal queues while both preemption modes keep rolling:

![Speed trace, 3x demand](img/counterfactual_ev_speed.png)

Reproduce: `python -m evaluation.counterfactual` (needs SUMO; 20 seeds ≈
40 min, or `--seeds 1-3` for a quick look).

---

## Is that statistically real?

With paired seeds (same traffic, baseline vs system), the time saved is
significant at every density — paired *t*-test *p* = 2.4×10⁻²⁰ / 1.4×10⁻¹¹ /
3.0×10⁻⁶, all with large-to-very-large effect sizes (Cohen's *dz* = 9.6 /
3.2 / 1.5). The civilian cost is significant-but-tiny at 1× (+0.8 s, 95% CI
[+0.1, +1.4]) and **not significant at 2× or 3×** — the green wave buys the
ambulance a minute for no detectable cost to other traffic at realistic
congestion. The 3× interval is wide ([44, 85] s), and that's stated plainly:
robust and significant everywhere, high-variance at 3×.

![Time saved with 95% confidence intervals](img/significance_ci.png)

Reproduce: `python -m evaluation.significance`.

### Is it the detection or the timing that matters?

A controlled run settles it. Re-running the *perfect-knowledge* mode at the
**same ~80 m trigger** the gated system uses (instead of 220 m) gives
63.1 / 56.5 / 67.6 s saved — and the closed-loop system, despite 15% audio
misses and noisy bearings, matches it: the gap is **not significant at any
density** (paired *p* = 0.48 / 0.16 / 0.43). What *is* significant is the
trigger distance — firing at 220 m instead of 80 m costs +10 s at 1×
(*p* = 0.016) and +25 s at 2× (*p* = 0.0015), because the fixed 12 s holds
expire before the ambulance reaches the far junctions.

So the honest takeaway is **trigger timing dominates detection accuracy**: a
noisy real detector firing at the right moment captures ~91–99% of what
flawless detection achieves, and firing too early throws benefit away. (An
earlier version of the writeup reported the closed-loop system "beating"
perfect knowledge — that was this same timing artifact, before the controlled
@80 m run isolated it.) The fixed hold is the next lever; demand-adaptive
timing is on the future-work list.

---

## How good does the detector have to be?

The closed-loop result assumes audio at 85% and vision at 90% per-tick hit
rate. How much do those numbers matter? `evaluation/sensitivity.py` sweeps
the per-tick hit rate of **both** modalities together from 95% down to 20%
(15 seeds each):

![Time saved vs detector hit-rate](img/sensitivity.png)

Time saved is **flat — 62–64 s across the entire 20–95% range, firing in 100%
of runs.** The system isn't sensitive to per-tick accuracy, and there's a
clean reason: the ambulance sits inside sensor range for ~50–100 fusion ticks
on approach, so even a 20%-per-tick detector almost certainly accumulates
enough to arm and confirm (1 − 0.8⁵⁰ ≈ 100%). The binding constraint isn't
frame accuracy — it's whether the vehicle is in range long enough, which the
150 m / 80 m ranges guarantee. This is the payoff of *temporal* fusion over
single-frame triggering: it absorbs an unreliable detector. (The real cliff
is below ~5% per-tick, far under any real detector.)

Reproduce: `python -m evaluation.sensitivity`.

---

## Does it stay quiet when there's no ambulance?

The counterfactual proves the system helps when an ambulance is present. It
says nothing about whether it stays quiet when one isn't — and a traffic
light that listens to the street is an attack surface. The three trust gates
(cross-modal cap, Doppler suppression, rate limit) exist to prevent false
green waves; `evaluation/adversarial.py` measures them, driving the **real
`TemporalFusionEngine`** through five scenarios × 30 seeds:

| scenario | what it injects | gate under test | peak belief | false fires |
|---|---|---|---|---|
| `phone_speaker` | loud on-corridor siren, **no camera** | cross-modal cap | 0.700 | **0** |
| `receding_ev` | departing ambulance, falling pitch | Doppler ×0.3 + cap | 0.700 | **0** |
| `cross_street` | siren ~90° off the corridor | bearing kernel | 0.005 | **0** |
| `noise_burst` | intermittent horn false alarms | arm-hold + decay | 0.700 | **0** |
| `spoof_flood` | attacker fakes **audio + vision** | state machine + rate limit | 1.000 | 1 (≤ 4 cap) |

**Zero false preemptions** in every benign scenario; the peak-belief column
*is* the proof — audio-only attacks pin to exactly the 0.700 cross-modal cap,
and the 90°-off siren never clears 0.005. The only thing that fires is a
*perfect* dual-modal spoof, which no system can tell from a real ambulance —
and even that's bounded twice over: the fusion state machine latches `ACTIVE`
after one fire, and the rate limiter caps re-fires at 4/lane/hour, all
hash-chain audited.

This runs as a **hard CI gate** (`tests/test_adversarial.py`) — the build
fails if any benign scenario ever false-fires. Reproduce:
`python -m evaluation.adversarial` (no SUMO; seconds).

### Versus a naive system

"Baseline vs ours" isn't a real comparison — the question is *ours vs the
naive method a simple system would use*. So the same scenarios run against a
**naive immediate-preemption baseline**: fire the green the instant any single
detection crosses threshold, with no cross-modal gate, no Doppler, no bearing,
no arm-hold, no rate limit — the classic "trigger on siren detected"
acoustic/optical EVP. 30 seeds per scenario:

| scenario | naive baseline | ours (gated) |
|---|---|---|
| phone_speaker (audio-only spoof) | 30 false fires | **0** |
| receding_ev (departing siren) | 30 | **0** |
| cross_street (90°-off siren) | 30 | **0** |
| noise_burst (intermittent horns) | **641** | **0** |
| spoof_flood (perfect dual-modal) | 30 | 30 *(both — undefendable)* |

The naive baseline false-fires in **all four** benign scenarios — 731 false
preemptions across 150 short runs, 641 of them from car horns alone. Ours
fires zero, only on the perfect dual-modal spoof. That gap is the contribution
of the trust gates: the difference between a deployable system and one any
phone speaker can hijack. When a real ambulance *is* present the two are
comparable (both fire), so the gates cost nothing in the true-positive case.

Reproduce: `python -m evaluation.adversarial --compare`.

---

## What if the ambulance turns off the corridor?

The route predictor assumes the ambulance follows the mapped corridor. When it
doesn't, the green wave has already launched for the *whole* corridor, so the
abandoned downstream junctions are held green for a vehicle that never shows.
`evaluation/route_uncertainty.py` runs the real pipeline twice per seed (8
seeds): a control where the ambulance finishes the corridor, and a deviation
where it's rerouted off right after the first signal.

| | result |
|---|---|
| deviation actually took effect | **8/8 seeds** |
| every preemption restored by end | **8/8** — no stuck greens |
| cost of a wrong route (civilian delay) | **+0.16 s** mean, +1.4 s worst |

The safety invariant holds: because each preemption holds a fixed 12 s then
restores, a wrong route **self-heals** — the abandoned junctions release on
schedule and the network is never stranded on green. The cost of guessing the
route wrong is statistically nil. Reproduce:
`python -m evaluation.route_uncertainty`.

---

## Does it generalize to another city?

Every other number comes from one intersection. So I built a **second real
network from OpenStreetMap — Shollinganallur, Chennai** — with a reproducible
pipeline (`osmGet.py` → `netconvert --tls.guess` → a shortest-path corridor
finder → `randomTrips.py`, all under `sim/nets/shollinganallur/`), then
pointed the *unchanged* `evaluation.counterfactual.simulate()` at it.
Generalization has two parts, and they came out differently:

- **Infrastructure — yes.** The entire pipeline runs unmodified on a
  completely different network: 10 signals, 540 background vehicles, a
  corridor it's never seen. Zero code changes.
- **Efficacy — corridor-dependent, and here it didn't help.** On this corridor
  the closed-loop system saved **−5.5 s (−3.5%, *p* ≈ 0.05, n=12)** — a slight
  slowdown, not a speed-up (civilian delay actually fell −2.4 s).

That negative is reported, not hidden, because it's informative. The corridor
`netconvert` produced is a **dense junction cluster** — 10 signals in ~570 m,
some 30–50 m apart — where Benz Circle is an arterial with signals 150–800 m
apart. The fixed timings (2.5 s all-red clearance per junction, 12 s holds)
are tuned for arterial spacing; at ~50 m spacing the per-junction ETAs
collapse to ~2 s apart, the cascade fires almost simultaneously, and the
stacked all-red clearances cost about as much as the coordination saves.

**The method generalizes to arterial-class corridors; it needs
spacing-adaptive timing for dense clusters** — the same adaptive hold the 2×
counterfactual finding pointed at. Honest scope beats a cherry-picked
corridor. Reproduce: `python -m evaluation.second_corridor`.

---

## Does it run in real time?

Two separate questions (`evaluation/latency.py`, measured on the actual
trained models, RTX 4060 Laptop):

**Compute** — each component, median wall-clock per call vs its real-time
budget:

| component | per call | budget | headroom |
|---|---|---|---|
| audio (CRNN siren + GCC-PHAT bearing + Doppler) | 6.2 ms | 100 ms (10 Hz) | **16×** |
| vision (YOLOv11s @ 640) | 7.2 ms | 40 ms (25 fps) | **5.6×** |
| fusion (one 10 Hz tick) | 0.003 ms | 100 ms | ~30000× |

![Compute latency vs budget](img/latency.png)

Every stage clears its budget comfortably on a *laptop* GPU — the hardware is
nowhere near the bottleneck.

**Action delay** — siren onset → light change is dominated not by compute but
by the *deliberate* certainty gating: the lane has to arm, hold for 0.5 s,
cross the preempt threshold, and get a camera confirmation. In the demo that
lands ~5–6 s after onset, which is the point — the few-millisecond compute
budget is spent buying confidence, not fighting the clock. Reproduce:
`python evaluation/latency.py`.

---

## The detectors

### Siren CRNN (audio)

Trained on 4,828 three-second windows @ 16 kHz mixed from four free datasets —
[EVSS](https://www.kaggle.com/datasets/vishnu0399/emergency-vehicle-siren-sounds),
[sireNNet](https://data.mendeley.com/datasets/j4ydzzv4kb/1),
[LSSiren](https://figshare.com/articles/dataset/Large-Scale_Dataset_for_Emergency_Vehicle_Siren_and_Road_Noises/17560865)
and [UrbanSound8K](https://zenodo.org/records/1203745) hard negatives — plus
SNR-mixed synthetic sirens. The train/val split is at **source-recording
level** (no window leakage).

On the held-out validation set it scores a perfect AUC 1.000 — but that number
isn't worth much on its own: those clips are isolated, clean sirens vs
isolated noise, an easy task. The number that matters is the
deployment-realistic one. `audio/hard_eval.py` mixes the *same held-out siren
recordings* into fresh UrbanSound8K street noise (every slice used in training
removed) across a sweep of SNR — lower SNR being the distance proxy, since a
far siren is quiet against the traffic around the mic:

| condition | AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| clean held-out (ceiling) | 1.000 | 1.000 | 1.000 | — |
| +5 dB SNR (near) | 0.976 | 0.919 | 0.934 | 0.927 |
| 0 dB SNR | 0.941 | 0.903 | 0.761 | 0.826 |
| −5 dB SNR (far / heavy traffic) | 0.859 | 0.869 | 0.544 | 0.669 |
| **overall hard set** | **0.925** | 0.965 | 0.746 | 0.842 |

![Siren detection under street noise](img/audio_hard_eval.png)

It degrades gracefully — AUC only falls to 0.86 at −5 dB, where a distant
siren is genuinely buried in traffic. Precision stays high throughout (0.87–
0.92); what drops is recall at low SNR. That's the right failure mode here —
better to miss a far-off siren for a moment than to false-preempt, and a real
approach gives the fusion layer many windows of rising SNR to accumulate over,
so one −5 dB miss doesn't lose the vehicle. The honest 0.925 is a stronger
claim than the clean 1.000. Reproduce: `python audio/hard_eval.py`.

![CRNN training curves](results/audio_training_curves.png)

### YOLOv11s ambulance detector (vision)

Trained 80 epochs on the
[Roboflow ambulance dataset](https://universe.roboflow.com/srivalli-yada/ambulance-wmxl5/dataset/5)
(623 train / 141 val / 77 test images) — 12.4 min on an RTX 4060 Laptop GPU.

| split | mAP@50 | mAP@50-95 | Precision | Recall |
|---|---|---|---|---|
| validation | **0.810** | 0.583 | 0.866 | 0.751 |
| test (held-out) | **0.800** | 0.584 | 0.820 | 0.748 |

Inference runs at 4.8 ms/frame — roughly 200 FPS capable, far above the 25 FPS
target.

| | |
|---|---|
| ![PR curve](results/yolo_pr_curve.png) | ![Predictions](results/yolo_val_predictions.jpg) |

---

## A note on the network

OSM has no traffic-signal tags in the Benz Circle part of Vijayawada, so
signals were placed at the four corridor junctions with `netconvert
--tls.set` (`sim/build_corridor.py` traces the corridor and picks them).
Signal positions are therefore modelled, not surveyed — the same caveat
applies to the Shollinganallur network, where `--tls.guess` placed them.
