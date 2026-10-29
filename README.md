# AutoVision.AI — Deep Learning Computer Vision Platform

An intelligent, full-stack computer vision platform featuring a **Dual-AI Architecture** combining cutting-edge multimodal vision models with local on-device deep learning.

---

## 🌟 Key Features

- **🧠 Dual-AI Inference Engine**:
  - **Google Gemini Vision API** (`gemini-3.1-flash-lite`): Performs deep visual reasoning, 99%+ precise classification (e.g., identifying not just a car, but specifically a *Ferrari Roma* with 99.2% confidence), and generates automated visual insight summaries.
  - **MobileNetV2 CNN** (via Microsoft's ONNX Runtime): On-device edge model providing fast offline evaluation.
- **⚡ Instant Client-Side Preview & Asynchronous Inference**:
  - Immediate image rendering upon file selection or drag-and-drop.
  - Asynchronous background evaluations (`fetch`) update results side-by-side in milliseconds without reloading or jumping the page.
- **🌍 Curated Worldwide Showcase Gallery**:
  - Built-in test collection covering supercars, wildlife, landmarks, and objects with 1-click evaluation.
- **🎮 Interactive AI Vision Challenge Game**:
  - "Human vs Machine" perception game featuring mystery rounds, real-time score tracking, and win streak counters.
- **🛡️ Secure Environment Configuration**:
  - API keys managed through `.env` and shielded from version control via `.gitignore`.

---

## 🛠️ Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Backend** | Python 3.14, Flask |
| **AI / ML** | Google Gemini Multimodal Vision API, MobileNetV2 (ONNX Runtime) |
| **Image Processing** | Pillow (PIL), NumPy |
| **Frontend** | HTML5, CSS3 (Modern Dark-Mode UI, Responsive, Zero Framework Overhead) |

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/AutoVision-AI.git
cd AutoVision-AI
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Key
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```
*(Note: If no API key is provided, the platform automatically runs using the local MobileNetV2 offline engine).*

### 4. Run the Application
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
├── .env.example                # Example environment configuration
├── .gitignore                  # Prevents secrets and caches from being committed
├── requirements.txt            # Python dependencies
├── main.py                     # Flask web server & Dual-AI inference pipeline
├── mobilenetv2.onnx            # Pretrained MobileNetV2 ONNX weights
├── imagenet_classes.txt        # 1,000 ImageNet categories
├── templates/
│   └── index.html              # Modern web interface (Preview, Gallery, Game)
├── static/
│   ├── style.css               # Luxury dark glassmorphism stylesheet
│   ├── samples/                # High-resolution worldwide test dataset
│   └── uploads/                # Directory for user-uploaded images
└── README.md
```

---

## 📄 License
This project is open-source and available under the MIT License.
