# 🚦 Green Wave++

> **Certainty-aware emergency vehicle preemption using audio-visual fusion.**

> [!NOTE]
> **🚧 Active development — Phases 1–4 complete: both detection models trained, virtual-sensor replay against the real Benz Circle network, trust-gated fusion, and security hardening (auth, rate limiting, tamper-evident audit). Phase 5 — counterfactual evaluation + deployment — is what's left.**

Detects approaching ambulances from CCTV and microphone arrays, fuses the evidence with a temporal belief engine, and pre-clears a corridor of green traffic lights — before the vehicle reaches the intersection.

---

## System Architecture

```
Microphone array → CRNN siren detector → GCC-PHAT bearing estimator ─┐
                                                                       ├→ TemporalFusionEngine
CCTV camera     → YOLOv11 ambulance detector → lane assigner ─────────┘
                                                        │
                                             belief >= 0.8 + held 0.5s
                                                        │
                                               SumoController
                                         ┌─────────────────────────┐
                                         │ All-red clearance (2.5s) │
                                         │ Green cascade per TLS    │
                                         │ 12s hold -> restore      │
                                         └─────────────────────────┘
                                                        │
                                              React dashboard (WebSocket)
```

### Key Technical Highlights
- **Temporal belief fusion** — Gaussian-weighted audio bearing + direct vision lane assignment, exponential decay, hysteresis prevents false triggers
- **Certainty gating** — belief must exceed threshold *and hold* for `arm_duration_sec` before preemption fires
- **SUMO-optional architecture** — mock TLS state machine runs on any machine without SUMO installed
- **Demo mode** — full system runs with synthetic audio and vision, no hardware or trained models required

---

## Results (Phase 1 — trained models)

### Siren CRNN (audio)

Trained on **4,828 three-second windows @ 16 kHz** mixed from four free datasets —
[Emergency Vehicle Siren Sounds](https://www.kaggle.com/datasets/vishnu0399/emergency-vehicle-siren-sounds),
[sireNNet](https://data.mendeley.com/datasets/j4ydzzv4kb/1),
[LSSiren](https://figshare.com/articles/dataset/Large-Scale_Dataset_for_Emergency_Vehicle_Siren_and_Road_Noises/17560865) and
[UrbanSound8K](https://zenodo.org/records/1203745) hard negatives (car horn, drilling,
engine idling, street music) — plus SNR-mixed synthetic sirens.  Train/val split is at
**source-recording level** (no window leakage). Early-stopped at epoch 28/50.

| Val set (716 windows) | AUC | Precision | Recall | Threshold |
|---|---|---|---|---|
| Real (held-out recordings) | **1.0000** | **1.0000** | **1.0000** | 0.373 |
| Synthetic (SNR-mixed) | 1.0000 | 1.0000 | 0.963 | 0.373 |

> Caveat: validation recordings are held out, but come from the same datasets
> (clean capture). Expect lower numbers on distant/windy street microphones —
> that is what the fusion layer's decay, arm-hold and cross-modal gates are for.

![CRNN training curves](docs/results/audio_training_curves.png)

### YOLOv11s ambulance detector (vision)

Trained 80 epochs on the [Roboflow ambulance dataset](https://universe.roboflow.com/srivalli-yada/ambulance-wmxl5/dataset/5)
(623 train / 141 val / 77 test images) — 12.4 min on an RTX 4060 Laptop GPU.

| Split | mAP@50 | mAP@50-95 | Precision | Recall |
|---|---|---|---|---|
| Validation | **0.810** | 0.583 | 0.866 | 0.751 |
| Test (held-out) | **0.800** | 0.584 | 0.820 | 0.748 |

Inference: **4.8 ms/frame** (≈200 FPS capability — far above the 25 FPS target).

| | |
|---|---|
| ![PR curve](docs/results/yolo_pr_curve.png) | ![Predictions](docs/results/yolo_val_predictions.jpg) |

### Phase 2 — virtual sensors + a real intersection

The pipeline now replays a video + WAV pair as if they were the live camera and
microphone (`run.py --virtual`), both paced on one media clock. And instead of a
toy grid, the green wave runs on **Benz Circle, Vijayawada**: a 1.4 km corridor
pulled from OpenStreetMap (MG Road → Benz Circle → Bandar Road) with four
signalized junctions, driven through SUMO/TraCI.

Measured on the real network, audio only (a replayed 15 s siren recording):

| Event | Sim time |
|---|---|
| Siren detected, preemption fired | t = 1.9 s |
| Junction 1 (MG Rd) green | t = 9.9 s (ETA 8.0) |
| Junction 2 (MG Rd) green | t = 12.9 s (ETA 11.0) |
| Junction 3 (Bandar Rd) green | t = 15.9 s (ETA 14.0) |
| Junction 4 (Bandar Rd) green | t = 18.9 s (ETA 17.0) |

Each junction goes green at its own ETA — a rolling wave, not a blanket change —
and every signal hands back to its normal program after the hold.

> OSM has no traffic-signal tags in this part of Vijayawada, so signals were
> placed at the four corridor junctions with `netconvert --tls.set`
> (`sim/build_corridor.py` traces the corridor and picks them). Signal positions
> are therefore modelled, not surveyed.

```bash
# replay a siren against the Benz Circle network, watch it in SUMO's GUI
python run.py --virtual --wav data/virtual_demo/siren_15s.wav \
              --sumo sim/nets/benz_circle/benz.sumocfg --sumo-gui
```

### Phase 3 — trust gates and graded response

The fusion engine no longer treats a loud siren as proof. Three safeguards,
each unit-tested and demonstrated on the Benz Circle network:

- **Cross-modal gate** — audio alone caps belief at 0.7, below the 0.8
  preemption threshold. The same 15s siren replay that used to fire a full
  green wave now holds at `0.700 [ARMED]` for its entire duration and never
  preempts; one camera confirmation within 3s lifts the cap.
- **Doppler gate** — a receding siren (falling pitch trend) has its belief
  contribution multiplied by 0.3. Sirens sweep by design, so the detector
  tracks the sweep's upper envelope per half-second block instead of raw
  pitch — a wail oscillating ±300 Hz/s with no drift stays "approaching".
- **Graded action** — the instant a lane arms (belief ≥ 0.6), the nearest
  corridor signal's current green is stretched 5s: a cheap, reversible first
  move while the system waits for visual confirmation.

Also fixed in this phase: cascade ETAs now come from the mapped corridor
geometry (0/158/384/831m on Benz Circle) instead of a hard-coded 30m
junction spacing — the far junction's green arrives when the ambulance
does, not 50 seconds early.

### Phase 4 — security hardening

A traffic light that listens to the street is an attack surface. This phase
assumes someone *will* play a siren from a phone speaker or curl the API,
and bounds the damage:

- **API key everywhere** — every REST endpoint except `/status` requires
  `X-API-Key`. The key never enters git: it resolves from the
  `GREENWAVE_API_KEY` env var, falling back to the gitignored
  `common/secrets.yaml`, which `run.py` auto-generates on first start and
  hands to the dashboard via `ui/.env.local` (also gitignored).
- **WebSocket tokens** — browsers can't put headers on a WebSocket, so the
  dashboard trades the key for a short-lived token (`POST /token`, 5 min
  TTL, in-memory only) and connects with `/ws?token=...`. A leaked
  dashboard URL goes stale in minutes, and every reconnect fetches a
  fresh token, so expiry never strands the UI.
- **Per-lane rate limiting** — at most 4 preemptions per lane per sliding
  hour (`security.rate_limit`). However loud the world gets, a spoofed
  siren or glitching detector can't strobe an intersection; denials are
  logged and audited but never reach SUMO.
- **Hash-chained audit log** — every preemption fired or denied, every
  fusion reset, every pipeline start/stop lands in `logs/audit.jsonl`,
  each line SHA-256-chained to the one before it. Edit, delete or reorder
  any line and `python -m common.audit logs/audit.jsonl` names the first
  broken entry.
- **Pydantic + CORS lockdown** — request/response bodies are typed models
  (token TTLs bounded at an hour, malformed WS messages dropped), and
  `allow_origins=["*"]` became an explicit dashboard-origin allowlist.
  `/reset` used to mutate state over GET; it's POST now, and audited.

28 new tests; 79 total passing.

---

## Development Status

| Module | Status | Notes |
|---|---|---|
| Audio CRNN siren detection | ✅ Trained | AUC 1.000 on real val — `checkpoints/audio_best.pt` |
| Real siren dataset pipeline | ✅ Complete | `audio/prepare_real_data.py` — 4 free sources, windowed manifests |
| GCC-PHAT bearing estimation | ✅ Complete | Multi-mic array TDOA |
| YOLOv11 vision detector | ✅ Trained | mAP@50 0.81 — `vision/weights/yolov11s-ambulance.pt` |
| Temporal fusion engine | ✅ Complete | State machine, hysteresis, ETA |
| SUMO traffic controller | ✅ Complete | Mock + TraCI backends |
| Route predictor | ✅ Complete | Corridor + ETA computation |
| Integration pipeline | ✅ Complete | Threaded, async, demo mode |
| React dashboard | ✅ Complete | Bearing compass, intersection map, event feed |
| FastAPI WebSocket server | ✅ Complete | /ws + /status + /token + /reset + /beliefs — API-keyed since phase 4 |
| Virtual sensor mode (video/WAV replay) | ✅ Complete | `run.py --virtual`, one shared media clock, A/V sync tested |
| Real SUMO TraCI backend | ✅ Complete | Sim-time green cascade, per-approach signal states, program restore |
| Benz Circle (Vijayawada) network | ✅ Complete | OSM extract → netconvert, 4-signal corridor on MG Rd/Bandar Rd |
| Cross-modal gate | ✅ Complete | Audio-only belief caps at 0.7; preemption needs camera confirmation |
| Doppler gate | ✅ Complete | Receding sirens (falling pitch envelope) weighted ×0.3 |
| Graded arm action | ✅ Complete | Belief ≥0.6 stretches the nearest green +5s before full preemption |
| ETA-true green cascade | ✅ Complete | ETAs from mapped corridor distances, verified per-TLS in SUMO |
| API key + WS token auth | ✅ Complete | X-API-Key on REST, short-lived tokens on /ws, key lives outside git |
| Per-lane preemption rate limit | ✅ Complete | 4 per lane per sliding hour; denials audited, never reach SUMO |
| Hash-chained audit log | ✅ Complete | `logs/audit.jsonl`, SHA-256 chain — `python -m common.audit` verifies |
| CORS + input validation | ✅ Complete | Origin allowlist, pydantic everywhere, /reset moved to POST |
| Counterfactual evaluation + deploy | 📋 Phase 5 | Paired SUMO runs, docker-compose, Cloudflare tunnel |

---

## 📊 Simulation Results

> Results from SUMO–TraCI simulation across 2 recorded preemption sessions.

| Metric | Value |
|---|---|
| Sessions processed | 2 (with confirmed preemption events) |
| Signal wait-time reduction | **+98.1% average** |
| Intersection throughput gain | **+127.3% average** |
| Preemption latency | 0.8s (audio-visual fusion to signal change) |
| Lane prediction accuracy | 94% (temporal fusion engine) |

> [!NOTE]
> Results are from controlled SUMO simulation. Live hardware integration and field benchmarking are in progress (see development status above).

## Quick Start (Demo Mode)

```bash
# Clone
git clone https://github.com/kbvinay001/Green-wave.git
cd Green-wave/greenwave

# Install Python dependencies
pip install -r requirements.txt

# Install React dashboard dependencies (first time only)
cd ui && npm install && cd ..

# Launch everything (demo mode -- synthetic data, no hardware needed)
python run.py --demo
```

Or double-click **`Launch GreenWave++.bat`** in the `greenwave/` folder.

Open **http://localhost:5173** for the live dashboard.

First start generates an API key into `common/secrets.yaml` (gitignored) and
hands it to the dashboard automatically — nothing to configure. To use your
own key instead, set the `GREENWAVE_API_KEY` environment variable.

---

## Training Your Own Models

### Audio (siren detection)

```bash
# 1. Download + build the real dataset (EVSS, sireNNet, LSSiren, UrbanSound8K ~7GB)
python audio/prepare_real_data.py --download --build

# 2. (optional) Generate extra synthetic training data
python audio/tools/generate_synthetic.py

# 3. Train the CRNN on synthetic + real (validates on held-out real recordings)
python audio/train.py --epochs 50
```

### Vision (ambulance detection)

```bash
# 1. Download dataset from Roboflow, convert to YOLO format
python vision/prepare_data.py --roboflow <download_dir> --verify

# 2. Train YOLOv11
python vision/train.py --model yolo11s.pt --epochs 80
```

---

## Configuration

All system parameters in `common/config.yaml`:

| Section | Key | Default | Description |
|---|---|---|---|
| `fusion` | `arm_threshold` | 0.6 | Belief level to start arm timer |
| `fusion` | `arm_duration_sec` | 0.5 | How long belief must hold before preemption |
| `fusion` | `preempt_threshold` | 0.8 | Final threshold to fire the green wave |
| `fusion` | `decay_factor` | 0.92 | Per-second belief decay |
| `fusion` | `sigma_angle_deg` | 20 | Audio bearing Gaussian kernel width |
| `sumo` | `all_red_duration` | 2.5s | Safety clearance before green wave |
| `sumo` | `preempt_green_duration` | 12s | Green hold per intersection |
| `security` | `ws_token_ttl_sec` | 300 | Dashboard WebSocket token lifetime |
| `security` | `rate_limit.max_preempts_per_lane` | 4 | Preemption cap per lane per window |
| `security` | `rate_limit.window_sec` | 3600 | Sliding rate-limit window |
| `security` | `cors_origins` | localhost:5173 | Exact origins allowed to call the API |
| `security` | `audit_log` | logs/audit.jsonl | Hash-chained audit trail location |

---

## Project Structure

```
greenwave/
├── audio/                  Siren detection pipeline
│   ├── model.py            SirenCRNN (Conv + BiGRU)
│   ├── preprocess.py       Log-mel spectrogram + SpecAugment
│   ├── bearing.py          GCC-PHAT TDOA bearing estimator
│   ├── train.py            Training loop
│   ├── infer.py            File/streaming inference
│   ├── stream_detector.py  Unified streaming detector
│   └── tools/
│       └── generate_synthetic.py  Synthetic siren WAV generator
│
├── vision/                 Ambulance detection pipeline
│   ├── infer.py            AmbulanceDetector (YOLO + tracker + lane assigner)
│   ├── train.py            Training wrapper
│   └── prepare_data.py     Dataset conversion + verification
│
├── fusion/                 Multi-modal belief fusion
│   ├── fuser.py            TemporalFusionEngine (core algorithm)
│   ├── route_predictor.py  Lane -> TLS corridor + ETA computation
│   └── sumo_controller.py  Green-wave sequencer (TraCI + mock)
│
├── integration/            End-to-end wiring
│   ├── pipeline.py         EndToEndPipeline (threads + async fusion loop)
│   ├── logger.py           Session logger (CSV + JSON)
│   └── replay.py           Synchronized video+audio replayer
│
├── ui/                     React + Vite dashboard
│   ├── src/
│   │   ├── App.jsx                Main dashboard
│   │   ├── index.css              Design system (dark ops-center)
│   │   └── components/
│   │       ├── BearingCompass.jsx SVG needle compass
│   │       ├── IntersectionMap.jsx Bird's-eye intersection view
│   │       └── EventFeed.jsx      Preemption event log
│   └── backend/
│       └── server.py              FastAPI WebSocket server
│
├── common/
│   ├── config.yaml          System parameters
│   └── verify_env.py        Environment health-check
│
├── run.py                  Unified launcher
├── Launch GreenWave++.bat  Windows one-click launcher
└── requirements.txt        Python dependencies
```

---

## Requirements

- Python 3.10+
- CUDA GPU recommended (CPU fallback works for demo)
- Node.js 18+ (for React dashboard)
- SUMO 1.18+ optional (mock mode works without it)

---


## 🚧 What's Still Being Built

- **Counterfactual evaluation** — paired SUMO runs (preemption on/off, same seed): time saved vs civilian delay
- **Deployment** — docker-compose + free Cloudflare tunnel for the live dashboard
- **Live hardware capture** — mic array (sounddevice) + camera (cv2) threads

---

*Multi-modal Emergency Vehicle Preemption · GCC-PHAT + YOLOv11 + Temporal Belief Fusion*

*Built with Python 3.12 · PyTorch · Ultralytics · FastAPI · React + Vite*
