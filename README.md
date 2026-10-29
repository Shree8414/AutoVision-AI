<div align="center">

# 🌐 AutoVision.AI
### Autonomous Computer Vision & Visual Reasoning Platform

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue?logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Backend-Flask-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Google Gemini](https://img.shields.io/badge/Multimodal%20AI-Google%20Gemini%20Vision-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)
[![ONNX Runtime](https://img.shields.io/badge/Edge%20Inference-ONNX%20Runtime-005CED?logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/Shree8414/AutoVision-AI/pulls)

<p align="center">
  <b>A state-of-the-art visual perception web platform powered by a Dual-AI Inference Engine combining frontier cloud visual reasoning with high-performance on-device neural edge computing.</b>
</p>

[Explore Features](#-core-features) • [System Architecture](#-system-architecture) • [Engine Comparison](#-dual-ai-engine-comparison) • [Quickstart](#-quickstart-guide) • [Project Structure](#-project-structure)

</div>

---

## 🌟 Core Features

- 🧠 **Dual-AI Inference Engine**:
  - **Google Gemini Multimodal Vision API (`gemini-3.1-flash-lite`)**: Deep contextual visual understanding and reasoning. Delivers granular predictions (e.g., distinguishing a *Ferrari Roma* with 99.2% certainty rather than a generic "sports car") along with automated visual analysis insights.
  - **MobileNetV2 Edge CNN (via Microsoft ONNX Runtime)**: High-speed, local neural network that executes entirely on-device with zero cloud latency and 100% offline autonomy.
- ⚡ **Zero-Latency In-Place UI Updates**:
  - Instant client-side previews upon file selection or drag-and-drop.
  - Asynchronous background evaluation (`fetch`) renders rich prediction cards side-by-side with zero page reload or jumping.
- 🌍 **Curated Worldwide Showcase Gallery**:
  - Preloaded high-resolution image collection spanning supercars, wildlife, aircraft, architecture, and daily objects for immediate 1-click evaluations.
- 🎮 **AI Perception Challenge Game**:
  - Interactive "Human vs Machine" perception contest featuring hidden mystery rounds, live scoreboards, and dynamic win streak tracking.
- 🛡️ **Production-Grade Secret Isolation**:
  - Cloud credentials and API keys are stored in strictly isolated `.env` configurations and guarded from version control via `.gitignore`.

---

## 🏗️ System Architecture

```text
 ┌────────────────────────────────────────────────────────────────────────┐
 │                              User Browser                              │
 │   • Drag-and-Drop Image Uploader       • Instant Client Image Preview  │
 │   • 1-Click Worldwide Gallery          • AI Perception Challenge Game  │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │ (Multipart POST / Asynchronous Fetch)
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                      Flask Web Application Server                      │
 │                   (d:/Image_Classification_Project)                    │
 └───────────────────┬────────────────────────────────┬───────────────────┘
                     │                                │
                     ▼ (Cloud Vision Request)         ▼ (On-Device Local Inference)
     ┌───────────────────────────────┐  Fallback  ┌───────────────────────────────┐
     │      Google Gemini API        │ ─────────► │      ONNX Runtime Engine      │
     │     gemini-3.1-flash-lite     │   offline  │      MobileNetV2 (14.2 MB)    │
     │  Deep Multimodal Reasoning   │            │   1,000 ImageNet Categories   │
     └───────────────┬───────────────┘            └───────────────┬───────────────┘
                     │                                            │
                     └────────────────────┬───────────────────────┘
                                          ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                      JSON Telemetry & Diagnostics                      │
 │      • Predicted Class        • Confidence Score (%)                   │
 │      • Inference Engine ID    • Visual Analysis Summary                │
 └────────────────────────────────────────────────────────────────────────┘
```

---

## ⚖️ Dual-AI Engine Comparison

| Capability | ☁️ Google Gemini Multimodal Vision | 💻 On-Device MobileNetV2 (ONNX) |
| :--- | :--- | :--- |
| **Model** | `gemini-3.1-flash-lite` | MobileNetV2 Pretrained CNN |
| **Execution Environment** | Google Cloud AI Infrastructure | Local CPU / Edge On-Device |
| **Internet Requirement** | Required | **None** (100% Offline Capable) |
| **Recognition Granularity** | Make, model, species, context (e.g. *Ferrari Roma*) | Broad category (e.g. *Sports Car*) |
| **Inference Output** | Prediction, Confidence %, Contextual Description | Top Predicted Class, Confidence % |
| **Average Latency** | ~0.8 – 1.2s | **~15 – 45ms** |
| **Fallback Role** | Primary High-Precision Engine | Automatic Offline Fallback |

---

## 🛠️ Technology Stack

- **Backend Framework**: Python 3.10+, Flask, Gunicorn / WSGI compatible
- **Cloud Multimodal AI**: Google Gemini API (`google-genai` / `gemini-3.1-flash-lite`)
- **Edge Inference Engine**: Microsoft ONNX Runtime (`onnxruntime 1.20+`)
- **Image Processing**: Pillow (PIL), NumPy
- **Frontend Architecture**: Modern HTML5, Responsive CSS3 (Glassmorphism Dark Theme, Vanilla JavaScript)

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/Shree8414/AutoVision-AI.git
cd AutoVision-AI
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy the template configuration file:
```bash
cp .env.example .env
```
Open `.env` in any text editor and add your Google Gemini API key:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```
> **Note**: If `GEMINI_API_KEY` is not provided or the network is disconnected, AutoVision.AI will gracefully and automatically run using the local on-device MobileNetV2 ONNX engine without interrupting your workflow!

### 5. Launch the Platform
```bash
python main.py
```
Open your browser and navigate to:
```text
http://127.0.0.1:5000
```

---

## 📁 Project Structure

```text
AutoVision-AI/
├── .env.example            # Environment variables template (safe for VCS)
├── .gitignore              # Shields .env, caches, and uploads from git
├── README.md               # Project documentation and architectural overview
├── requirements.txt        # Pinned Python package dependencies
├── main.py                 # Core Flask backend, routing & Dual-AI inference pipeline
├── mobilenetv2.onnx        # Pretrained MobileNetV2 ONNX neural network weights
├── imagenet_classes.txt    # 1,000 standard ImageNet classification labels
├── templates/
│   └── index.html          # Dynamic UI (Preview, Asynchronous Results, Game, Gallery)
├── static/
│   ├── style.css           # Glassmorphism dark luxury styling & responsive layout
│   └── samples/            # Curated worldwide test dataset (cars, animals, objects)
└── images/                 # Reference test images
```

---

## 🛡️ Security & Privacy

- **Zero Secret Leakage**: The production `.env` file holding secret API keys is explicitly excluded by `.gitignore` and has never been committed into the Git history.
- **Local Fallback Mode**: The platform remains completely functional offline using Microsoft's ONNX runtime, ensuring data privacy when cloud communication is disabled.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
Feel free to check the [issues page](https://github.com/Shree8414/AutoVision-AI/issues) or submit a Pull Request.

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for more information.

<div align="center">
  <sub>Engineered with ❤️ by <a href="https://github.com/Shree8414">Shree8414</a></sub>
</div>
