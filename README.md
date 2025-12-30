## LLM-Guided Multi-Drone Coordination for Simultaneous Multi-Accident Detection and Scene Understanding
## 📌 Overview

Rapid and effective response to highway accidents is critical for minimizing injuries and saving lives.
This repository presents an end-to-end, LLM-guided multi-drone incident management framework that integrates natural language understanding, real-time visual perception, vision–language scene understanding, and autonomous UAV coordination, validated in the Gazebo simulation environment.

The system enables simultaneous monitoring and management of multiple accident sites using a fleet of autonomous drones, guided by large language models and multimodal AI components.

![abstractfig1](https://github.com/user-attachments/assets/234f1a9b-9fd1-4960-a7d5-d669705edebf)

## 🚁 System Architecture

The proposed framework integrates the following key components:

* GPT-4o mini (LLM)
Parses simulated emergency call transcripts to extract structured information such as geographic coordinates and incident details.

* Multi-Agent UAV Controller
Dynamically assigns up to four autonomous drones to distinct accident locations.

* ArduPilot + MAVROS
Manages UAV flight control and navigation under GUIDED mode within Gazebo.

* YOLOv11n Object Detector
Custom-trained for real-time detection of highway accidents and fire-related events using curated Roboflow datasets.

* BLIP-2 Vision–Language Model (Fine-Tuned)
Generates structured, human-readable scene summaries from aerial imagery.

* Piper TTS
Converts generated scene descriptions into speech for audio-based situational awareness.

## 🧠 Key Contributions

* End-to-end LLM-guided incident management pipeline for autonomous emergency response

* Multi-drone coordination for simultaneous monitoring of multiple accident scenes

* Real-time visual detection + vision–language scene understanding from aerial imagery

* Fine-tuning of BLIP-2 on 1,783 image–caption pairs with frozen visual encoder

* Quantitative evaluation using BLEU-4, ROUGE-L, METEOR, CIDEr, and SPICE metrics

* Modular, scalable architecture validated in Gazebo simulation

## 📊 Evaluation

The vision–language scene summarization module was evaluated on a held-out test set of 254 samples using standard image-captioning metrics:

* BLEU-4

* ROUGE-L

* METEOR

* CIDEr

* SPICE

Results demonstrate robust scene understanding suitable for real-time emergency response scenarios.

## 🧪 Simulation Environment

* Simulator: Gazebo

* UAV Stack: ArduPilot SITL + MAVROS

* Number of UAVs: Up to 4 autonomous drones

* Sensor Input: Live aerial RGB imagery


## ⚙️ Requirements

* Ubuntu 20.04 / 22.04

* ROS (Noetic recommended)

* Gazebo

* ArduPilot + MAVROS

* Python ≥ 3.8

* PyTorch

* Ultralytics YOLO

* Hugging Face Transformers

* Piper TTS
 

## 🚀 Getting Started (High-Level)

* Set up the ROS + Gazebo + ArduPilot simulation environment

* Launch the multi-UAV simulation in Gazebo

* Start the LLM node to process emergency call transcripts

* Run YOLOv11n inference on live aerial streams

* Generate scene summaries using the fine-tuned BLIP-2 model

* Enable audio feedback via Piper TTS

## 📌 Applications

* Autonomous emergency response systems

* Intelligent transportation systems (ITS)

* Multi-UAV coordination research

* Vision–language navigation and reasoning

* Disaster monitoring and smart cities

## 🤝 Contact

* For questions, collaboration, or issues, please contact the authors.
