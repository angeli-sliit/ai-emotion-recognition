
# 🎭 AI Emotion Recognition Web Application
🚀 Live Demo: https://visionlense.streamlit.app/
> **A modern, real-time facial emotion detection app powered by Deep Learning**  
> Built using **TensorFlow, OpenCV, and Streamlit** with an **ultra-premium UI**, live webcam support, and Docker-ready deployment.

---

## 🚀 Key Features

| Category | Highlights |
|----------|------------|
| 🎥 Live Detection | Real-time webcam-based emotion analysis |
| 📷 Image Upload | Upload photos for accurate inference |
| 📊 Confidence Metrics | Detailed probability scores for all emotions |
| 🎨 Modern UI/UX | Gradient backgrounds, glassmorphism, smooth animations |
| 💻 Responsive Design | Fully optimized for desktop and mobile |
| 🐳 Docker Support | One-click deployment using Docker containers |

---

## 🎨 UI Enhancements

- Eye-catching **hero section** with modern typography  
- **Glassmorphic cards**, soft shadows, and animated effects  
- **Smooth hover & fade-in transitions**  
- **Color-coded emotions** for faster interpretation  
- **Gradient-themed dark UI**, fully customized with CSS

| Emotion | Emoji | UI Color |
|---------|-------|----------|
| Angry | 😠 | 🔴 Red |
| Disgust | 🤢 | 🟤 Dark Brown |
| Fear | 😨 | 🟣 Purple |
| Happy | 😊 | 🟡 Yellow |
| Neutral | 😐 | ⚪ Gray |
| Sad | 😢 | 🔵 Blue |
| Surprise | 😲 | 🟠 Orange |

---

## 🧠 Tech Stack

- **Python, Streamlit**
- **TensorFlow, Keras (MobileNetV2)**
- **OpenCV (Haar Cascade)**
- **Docker**
- **PIL, NumPy**

---

## 📁 Project Structure

```
facial_emotion/
│
├── webapp/
│   ├── webapp.py             # Main Streamlit Web App
│   ├── Dockerfile            # Docker config
│   ├── docker-compose.yml    # Docker Compose setup
│   ├── requirements.txt      # Dependencies
│   ├── run.sh                # Quick launch (Linux/Mac)
│   └── run.bat               # Quick launch (Windows)
│
├── mod_my_model01.keras      # Trained model (parent directory)
├── face022.ipynb             # Training notebook
│
├── train/
│   ├── 0/ ... 6/             # Emotion class folders
├── test/
│   ├── 0/ ... 6/
```

---

## 📦 Prerequisites

```
✓ Python 3.10+
✓ Model file: mod_my_model01.keras
✓ Webcam (optional)
✓ Docker (optional)
```

---

## ⚙️ Installation & Quick Start

### 🔹 Option 1 – Local Setup

```bash
cd webapp
pip install -r requirements.txt
streamlit run webapp.py
# Access at http://localhost:8501
```

### 🔹 Option 2 – Quick Script Launch

```bash
# Linux / Mac
cd webapp
chmod +x run.sh
./run.sh

# Windows
cd webapp
run.bat
```

### 🔹 Option 3 – Deploy via Docker

```bash
cd webapp
docker-compose up -d
```

Or build manually:

```bash
docker build -f webapp/Dockerfile -t emotion-app .
docker run -p 8501:8501 -v $(pwd)/mod_my_model01.keras:/app/mod_my_model01.keras emotion-app
```

---

## 📖 Usage Guide

### 🖼️ Image Upload
1. Click **Image Upload**
2. Select an image file
3. Click **Detect Emotion**
4. View results and probability charts

### 🎥 Live Detection
1. Click **Start live detection**
2. Align your face
3. Real-time predictions with emoji & confidence visualization

---

## 🧪 Model Info

| Property | Value |
|----------|-------|
| Architecture | MobileNetV2 |
| Input Size | 224 × 224 |
| Classes | 7 emotions |
| Framework | TensorFlow/Keras |
| Dataset | FER2013 |

---

## 🛠 Troubleshooting

| Issue | Solution |
|------|----------|
| Model not found | Place `mod_my_model01.keras` correctly |
| Webcam not working | Check camera permissions |
| Port conflict | Use `-p 8502:8501` |
| Docker error | Increase memory or mount model correctly |

---

## ☁ Deployment Options

### 🔹 Heroku

```bash
echo "web: streamlit run webapp.py --server.port=$PORT --server.address=0.0.0.0" > Procfile
git push heroku main
```

### 🔹 AWS EC2

```bash
ssh -i key.pem ubuntu@your-ec2-ip
docker run -d -p 8501:8501 emotion-app
```

### 🔹 Google Cloud Run

```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/emotion-app
gcloud run deploy emotion-app --platform managed
```

---

## 📝 License

MIT License

---

## 🚀 Contributing

Contributions are welcome!  
Feel free to open an issue or pull request.

---

## 🙏 Acknowledgements

- FER2013 dataset
- TensorFlow team
- Streamlit team
- OpenCV team

> _Made with ❤️ using Streamlit and TensorFlow_
