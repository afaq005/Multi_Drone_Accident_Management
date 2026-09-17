
<div align="center">

# 🚁 Agentic Multi-Drone Framework for Autonomous Multi-Accident Detection, Reasoning, and Response

**An end-to-end, LLM-guided multi-drone incident management system** — natural language mission planning, autonomous UAV coordination, real-time perception, and vision–language scene reasoning, validated in Gazebo and on physical hardware.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](#-requirements)
[![ROS](https://img.shields.io/badge/ROS-Noetic-22314E.svg)](#-requirements)
[![Simulator](https://img.shields.io/badge/simulator-Gazebo-orange.svg)](#-simulation-environment)


<img width="3448" height="3196" alt="fig_m" src="https://github.com/user-attachments/assets/5656332e-f645-4611-819a-a6dc253d2f7b" />

*Figure: Emergency call → language planning → drone allocation → navigation → detection → scene reasoning → dispatch, in a closed loop.*

</div>

---

## 📖 Overview

Highway accidents cost roughly **1.19 million lives every year**. Traditional incident management — static cameras, human monitoring, manual dispatch — is slow, has limited coverage, and degrades in poor visibility.

This repository implements a **five-agent, closed-loop framework** where autonomous agents perceive, reason, and act together to manage **multiple simultaneous accidents** without step-by-step human supervision:

| # | Agent | Role | Backbone |
|---|-------|------|----------|
| 1 | **Planning Agent** | Parses emergency-call transcripts into geo-coordinates & mission intent | GPT-4o mini |
| 2 | **Coordination Agent** | Solves drone-to-incident assignment (min-cost matching) | Constrained optimization |
| 3 | **Perception Agent** ×N | Real-time accident/fire detection on live aerial video, one per drone | YOLOv11n (custom-trained) |
| 4 | **Description Agent** | Converts detections into structured natural-language scene reports, with confidence-weighted fusion across drones | Fine-tuned BLIP-2 |
| 5 | **Dispatch Agent** | Converts reports to audio and routes alerts to the nearest rescue center | Piper TTS |

Up to **four autonomous drones**, each flown via **ArduPilot + MAVROS** in `GUIDED` mode, are coordinated end-to-end — from a raw 911-style transcript to a geo-tagged, spoken alert at a rescue center — in about **7.7 seconds median latency**.

The system was validated in **Gazebo simulation** (four simultaneous accident sites) and on **physical hardware** using a DJI M30T drone with RGB + infrared cameras.

---

## 🏗️ System Architecture

<div align="center">
<img src="docs/assets/fig2_gazebo_pipeline.png" alt="Gazebo simulation pipeline" width="800"/>
</div>

```
Emergency Call Transcript
        │
        ▼
┌───────────────────┐        ┌────────────────────────┐
│  Planning Agent    │──────▶│  Coordination Agent      │
│  (GPT-4o mini)      │       │  (assignment solver)     │
└───────────────────┘        └────────────┬─────────────┘
                                           ▼
                          ┌────────────────────────────────┐
                          │  ArduPilot + MAVROS (GUIDED)     │
                          │  Drone 1 · Drone 2 · Drone 3 · 4  │
                          └────────────────┬───────────────┘
                                           ▼
                     ┌──────────────────────────────────────┐
                     │  Perception Agents (YOLOv11n, per drone) │
                     └────────────────────┬─────────────────┘
                                          ▼
                     ┌──────────────────────────────────────┐
                     │  Description Agent (Fine-tuned BLIP-2)   │
                     │  + confidence-weighted multi-view fusion │
                     └────────────────────┬─────────────────┘
                                          ▼
                     ┌──────────────────────────────────────┐
                     │  Dispatch Agent (Piper TTS + routing)    │
                     │  → nearest rescue center                 │
                     └──────────────────────────────────────┘
```

---

## 📂 Repository Structure

> **Note:** folder names below are proposed so the codebase mirrors the five-agent architecture in the paper. Rename existing folders as indicated ("current → new") when reorganizing.

```
Multi_Drone_Accident_Management/
│
├── planning_agent/            # GPT-4o mini transcript parsing → structured JSON (coords, intent)
│   ├── prompt_templates/
│   └── llm_waypoint_node.py
│
├── coordination_agent/        # Drone-to-incident assignment solver (Eq. 2), waypoint publishing
│   └── assignment_solver.py
│
├── perception_agent/          # ⬅ currently "YoloDetection"
│   ├── train_yolov11n.py
│   ├── weights/                # link/checkpoint, not committed raw
│   └── inference_node.py
│
├── description_agent/         # ⬅ currently "blip_finetuning"
│   ├── finetune_blip2.py
│   ├── fusion.py                # confidence-weighted multi-view caption fusion (Eq. 23)
│   └── inference.py
│
├── dispatch_agent/             # NOT YET IN REPO — Piper TTS + nearest-rescue-center routing
│   ├── tts_dispatch.py
│   └── rescue_center_router.py
│
├── simulation/                 # NOT YET IN REPO — Gazebo worlds + ArduPilot/MAVROS bridge
│   ├── worlds/                  # 4-accident-site Gazebo world
│   ├── launch/                  # sim_vehicle.py / MAVROS launch files
│   └── gnc_controller/          # per-drone GNC flight controller node
│
├── dataset_finetuning/         # Dataset curation & augmentation scripts (Roboflow compilation)
│
├── video_inference/            # Real-world video testing (YOLOv11n + BLIP-2/SmolVLM comparison)
│
├── model/                       # Shared model utilities / configs
│
├── docs/
│   └── assets/                  # README images (see below — replace expiring GitHub links)
│
├── requirements.txt             # NOT YET IN REPO
├── environment.yml              # optional conda alternative
├── LICENSE                      # NOT YET IN REPO
└── README.md
```

**Priority additions** (currently missing from the repo but central to the paper):
- `planning_agent/` — GPT-4o mini prompt + JSON parsing logic (§6.2)
- `coordination_agent/` — the assignment-problem solver (Eq. 2)
- `dispatch_agent/` — Piper TTS + nearest-rescue-center routing (Eq. 24–25)
- `simulation/` — Gazebo world files and the ArduPilot/MAVROS launch stack (§6.1, §6.3)

Without these, the repo currently reflects only the perception + description sub-components, not the full agentic pipeline the paper describes.

---

## ⚙️ Requirements

| Component | Version / Notes |
|---|---|
| OS | Ubuntu 20.04 / 22.04 |
| ROS | Noetic (recommended) |
| Simulator | Gazebo |
| Flight stack | ArduPilot SITL + MAVROS |
| Python | ≥ 3.8 |
| Deep learning | PyTorch, Ultralytics YOLO, Hugging Face Transformers |
| TTS | Piper TTS |
| LLM API | OpenAI API key (GPT-4o mini) |

Install Python dependencies:

```bash
git clone https://github.com/afaq005/Multi_Drone_Accident_Management.git
cd Multi_Drone_Accident_Management
pip install -r requirements.txt
```

> `requirements.txt` is not yet in the repo — add one pinning at least: `ultralytics`, `transformers`, `torch`, `openai`, `piper-tts`, `rospy` (or note that ROS packages are installed via apt/rosdep, not pip).

---

## 🚀 Getting Started

1. **Set up the simulation stack**
   ```bash
   # ROS + Gazebo + ArduPilot SITL + MAVROS
   # (see simulation/README.md once added)
   ```
2. **Launch the multi-UAV Gazebo world** (4 drones, 4 accident sites)
3. **Start the planning agent** to process an emergency-call transcript:
   ```bash
   export OPENAI_API_KEY=your_key_here
   python planning_agent/llm_waypoint_node.py
   ```
4. **Run the coordination agent** to resolve drone-to-incident allocation
5. **Run per-drone YOLOv11n inference** on live aerial streams:
   ```bash
   python perception_agent/inference_node.py --drone-id 1
   ```
6. **Generate scene summaries** with the fine-tuned BLIP-2 description agent:
   ```bash
   python description_agent/inference.py --frame path/to/frame.jpg
   ```
7. **Enable audio dispatch** via Piper TTS:
   ```bash
   python dispatch_agent/tts_dispatch.py
   ```

> Each command above assumes the proposed folder structure — update paths to match the current repo layout until reorganized.

---

## 🧠 Key Contributions

- 🗣️ **Natural-language mission planning** — GPT-4o mini extracts incident coordinates and intent directly from free-form emergency-call transcripts.
- 🎯 **Hybrid dispatch** — supports both fully autonomous (optimization-based) and operator-directed drone-to-incident assignment.
- 👁️ **Real-time per-drone perception** — custom YOLOv11n detector (2 classes: Accident, Fire) trained on an augmented, curated Roboflow dataset.
- 📝 **Fine-tuned BLIP-2 scene description** — 1,783 image–caption training pairs, frozen visual encoder, with confidence-weighted fusion when multiple drones observe the same incident.
- 🔊 **Autonomous dispatch** — Piper TTS + nearest-rescue-center routing, with no human in the loop.
- 🏗️ **Modular, agentic architecture** — agents communicate over ROS/MQTT, so any component (LLM, vision backbone, control stack) can be swapped independently.
- 🌍 **Simulation-to-real validation** — tested in Gazebo (4 simultaneous incidents) and physically with a DJI M30T drone (RGB + infrared).

---

## 📊 Evaluation Results

### Perception Agent — YOLO model comparison (test set)

| Model | P | R | mAP@50 | Inference (ms) | Params (M) | GFLOPs |
|---|---|---|---|---|---|---|
| YOLOv12n | 0.830 | 0.770 | 0.841 | 3.1 | 2.55 | 6.3 |
| **YOLOv11n** ✅ | 0.813 | **0.826** | **0.874** | 2.2 | 2.58 | 6.3 |
| YOLOv10n | **0.886** | 0.722 | 0.852 | 2.7 | 2.69 | 8.2 |
| YOLOv9t | 0.824 | 0.777 | 0.841 | 2.8 | 1.97 | 7.6 |
| YOLOv8n | 0.830 | 0.803 | 0.860 | 1.9 | 3.01 | 8.1 |
| YOLOv6n | 0.857 | 0.755 | 0.848 | **1.8** | 4.23 | 11.8 |
| YOLOv5n | 0.863 | 0.803 | 0.859 | 2.4 | 2.50 | 7.1 |

**YOLOv11n** was selected for the framework — best recall and mAP@50 among nano variants at competitive compute cost.

### Description Agent — Fine-tuned BLIP-2 (254 held-out test pairs)

| Model | BLEU-4 | ROUGE-L | METEOR | CIDEr | SPICE |
|---|---|---|---|---|---|
| BLIP-2 (fine-tuned) | 11.70 | 30.12 | 35.05 | 45.05 | 21.42 |

### Inference latency: BLIP-2 (fine-tuned) vs. SmolVLM

| Scenario | SmolVLM (s) | BLIP-2 FT (s) |
|---|---|---|
| Accident Case 1 | 3.63 | **1.413** |
| Accident Case 2 | 3.59 | **1.349** |
| Fire Case 1 | 3.27 | **1.912** |
| Fire Case 2 | 3.76 | **1.413** |

### End-to-end pipeline latency (median, per stage)

| Stage | Median Time (ms) |
|---|---|
| GPT-4o mini waypoint extraction | 620 |
| ROS topic propagation | 30 |
| Take-off & cruise to first waypoint | 5,900 |
| YOLO inference (640×480, RTX 3090 FP16) | 38 |
| BLIP-2 fine-tuned caption | 1,100 |
| MQTT push to server | 8 |
| **Total end-to-end** | **~7,700 (7.7 s)** |

> Latency figures were measured on an RTX 3090; results will vary with different hardware.

---

## 🎥 Real-World & Physical Validation

- **YouTube video testing**: 4 real-world videos (2 accident, 2 fire) — YOLOv11n detection followed by BLIP-2 fine-tuned captioning, benchmarked against SmolVLM.
- **Physical outdoor testing**: DJI M30T drone, RGB + infrared, two physically staged collision scenarios — confirms perception and description agents generalize from simulation to real sensing hardware without architectural changes.

<div align="center">
<img src="docs/assets/fig8_physical_testing.png" alt="Physical M30T drone testing" width="800"/>
</div>

---

## 📦 Datasets & Pretrained Weights

| Asset | Status | Notes |
|---|---|---|
| YOLO detection dataset (Accident/Fire, 2,548 images → 4,331 augmented) | Compiled from 2 public Roboflow datasets + prior study dataset | Add links to [Roboflow dataset 1](https://universe.roboflow.com/donghee/test-d95ea), [Roboflow dataset 2](https://universe.roboflow.com/kk-qg4vu/car-fires-detection) |
| BLIP-2 fine-tuning dataset (1,783 train / 511 val / 254 test image–caption pairs) | Generated via moondream2 captioning | Available upon request — add contact/request process here |
| YOLOv11n trained weights | **Not yet linked** | Add a GitHub Release or Hugging Face Hub link |
| Fine-tuned BLIP-2 checkpoint (`Salesforce/blip2-opt-2.7b` base) | **Not yet linked** | Add a Hugging Face Hub link |

---

## 🖼️ Fixing the README Images

The current README embeds a `private-user-images.githubusercontent.com` link containing a **short-lived signed JWT** — it will expire and break. Replace it by:

```bash
mkdir -p docs/assets
# move your local copies of the figures here, e.g.:
# docs/assets/fig1_framework_overview.png
# docs/assets/fig2_gazebo_pipeline.png
# docs/assets/fig8_physical_testing.png
git add docs/assets
```

Then reference them with relative paths, exactly as done in this README (`docs/assets/...`), so images render permanently regardless of upload session.

---

## 🔭 Roadmap / Limitations (from the paper)

- [ ] Robustness testing under night, fog, and rain conditions
- [ ] Reduce onboard inference latency (e.g., mask token distillation)
- [ ] Temporal cross-attention for consistent multi-frame scene summaries
- [ ] Extend coordination agent to drone failure / dynamic mid-mission re-allocation and incident counts exceeding available drones
- [ ] Improve planning agent's ability to infer allocation from ambiguous or partially specified transcripts (currently relies on explicit per-drone assignment when given)
- [ ] Speech-aware language models to compensate for ASR errors in noisy emergency-call audio

---

## 📄 Citation

```bibtex
@article{ahmed2026agentic,
  title   = {An Agentic Multi-Drone Framework for Autonomous Multi-Accident Detection, Reasoning, and Response},
  author  = {Ahmed, Afaq and Eesaar, Hassan and Farhan, Muhammad and Yoo, YongSuk and Lee, Deok Jin},
  journal = {[add venue / arXiv ID once available]},
  year    = {2026}
}
```

---

## 🤝 Contributing

Contributions are welcome — especially for the missing `planning_agent/`, `coordination_agent/`, `dispatch_agent/`, and `simulation/` modules described above. Please open an issue before submitting a large PR so the scope can be discussed.

---

## 📬 Contact

For questions, collaboration, or issues, please contact the authors:
📧 **afaq@jbnu.ac.kr**

Affiliated with the **Center for Autonomous Intelligence and e-Mobility**, Jeonbuk National University.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE) — *(update to match your actual chosen license; none is currently set in the repo)*.

<div align="center">

*Built with 🛰️ ROS · 🐍 Python · 🤖 PyTorch · 🦾 ArduPilot*

</div>
