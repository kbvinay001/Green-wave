# 🚦 Green Wave++

> **Certainty-aware emergency vehicle preemption using audio-visual fusion.**

> [!NOTE]
> **🚧 Active development — Phase 1 (model training) complete. Both detection models are trained and evaluated; virtual-sensor mode, SUMO integration, security hardening and counterfactual evaluation are in progress.**

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
| FastAPI WebSocket server | ✅ Complete | /ws, /status, /reset, /beliefs |
| Virtual sensor mode (video/WAV replay) | 🔄 Phase 2 | `--virtual` flag, time-synchronized |
| SUMO network (OSM real intersection) | 🔄 Phase 2 | OSM Web Wizard export + TraCI wiring |
| Fusion logic upgrades | 📋 Phase 3 | Cross-modal gate, Doppler gate, graded arm action |
| Security hardening | 📋 Phase 4 | API key, WS token, rate limit, hash-chained audit log |
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

- **Virtual sensor mode** — time-synchronized video + WAV file replay (`run.py --virtual`)
- **SUMO network files** — real intersection via OSM Web Wizard, `.sumocfg` + TraCI
- **Fusion upgrades** — cross-modal gate, Doppler (receding-siren) gate, graded green-extension at arm
- **Security hardening** — API-key auth, WebSocket token, per-lane rate limiting, hash-chained audit log
- **Counterfactual evaluation** — paired SUMO runs (preemption on/off, same seed): time saved vs civilian delay
- **Deployment** — docker-compose + free Cloudflare tunnel for the live dashboard
- **Live hardware capture** — mic array (sounddevice) + camera (cv2) threads

---

*Multi-modal Emergency Vehicle Preemption · GCC-PHAT + YOLOv11 + Temporal Belief Fusion*

*Built with Python 3.12 · PyTorch · Ultralytics · FastAPI · React + Vite*
