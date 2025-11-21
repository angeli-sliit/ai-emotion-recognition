# AI Emotion Recognition Web Application

A modern, responsive web application for real-time facial emotion detection using deep learning. Built with Streamlit and TensorFlow.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)

## 🌟 Features

- **Modern UI/UX**: Clean, minimalistic design with gradient backgrounds and smooth animations
- **Image Upload**: Upload photos to detect emotions
- **Live Webcam**: Real-time emotion detection using your webcam
- **Detailed Analysis**: View confidence scores and all emotion predictions
- **Responsive Design**: Works seamlessly on desktop and mobile devices
- **Docker Support**: Easy deployment with Docker containerization

## 🎨 UI Features

- **Hero Section**: Eye-catching header with clear call-to-action buttons
- **Gradient Backgrounds**: Beautiful blue-purple gradient theme
- **Glassmorphism Effects**: Modern UI elements with soft shadows
- **Smooth Animations**: Fade-in effects and hover interactions
- **Color-Coded Emotions**: Each emotion has its unique color
  - 😠 Angry → Red
  - 🤢 Disgust → Dark Brown
  - 😨 Fear → Purple
  - 😊 Happy → Yellow
  - 😐 Neutral → Gray
  - 😢 Sad → Blue
  - 😲 Surprise → Orange

## 📋 Prerequisites

- Python 3.10 or higher
- Trained model file (`mod_my_model01.keras`) in parent directory
- Webcam (optional, for live detection)
- Docker (optional, for containerized deployment)

## 🚀 Quick Start

### Option 1: Local Installation

1. **Navigate to webapp directory**
   ```bash
   cd webapp
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure model file exists**
   - Make sure `mod_my_model01.keras` is in the parent directory (`../mod_my_model01.keras`)
   - If not, train the model using `face022.ipynb` in the parent directory

4. **Run the application**
   ```bash
   streamlit run webapp.py
   ```

5. **Access the app**
   - Open your browser and navigate to `http://localhost:8501`

### Option 2: Using Quick Start Scripts

**Linux/Mac:**
```bash
cd webapp
chmod +x run.sh
./run.sh
```

**Windows:**
```bash
cd webapp
run.bat
```

### Option 3: Docker Deployment

1. **Navigate to webapp directory**
   ```bash
   cd webapp
   ```

2. **Build and run with Docker Compose**
   ```bash
   docker-compose up -d
   ```

3. **Or build Docker image manually**
   ```bash
   # From parent directory
   docker build -f webapp/Dockerfile -t emotion-app .
   docker run -p 8501:8501 -v $(pwd)/mod_my_model01.keras:/app/mod_my_model01.keras emotion-app
   ```

4. **Access the app**
   - Open your browser and navigate to `http://localhost:8501`

## 📖 Usage Guide

### Upload Photo

1. Click on **"Upload Photo"** button or use the file uploader
2. Select an image file (JPG, JPEG, or PNG)
3. Wait for face detection
4. Click **"Detect Emotion"** to analyze
5. View the results with confidence scores

### Live Webcam

1. Click **"Start Webcam"** button
2. Position your face in front of the camera
3. Click **"Capture Photo"** when ready
4. Click **"Analyze Captured Image"** to detect emotion
5. View detailed results

## 🏗️ Project Structure

```
facial_emotion/
│
├── webapp/                  # Web application directory
│   ├── webapp.py           # Main Streamlit application
│   ├── Dockerfile          # Docker configuration
│   ├── docker-compose.yml  # Docker Compose configuration
│   ├── requirements.txt    # Python dependencies
│   ├── README.md           # This file
│   ├── run.sh              # Quick start script (Linux/Mac)
│   └── run.bat             # Quick start script (Windows)
│
├── mod_my_model01.keras    # Trained model file (parent directory)
├── face022.ipynb           # Model training notebook
│
├── train/                   # Training dataset
│   ├── 0/                  # Angry
│   ├── 1/                  # Disgust
│   ├── 2/                  # Fear
│   ├── 3/                  # Happy
│   ├── 4/                  # Neutral
│   ├── 5/                  # Sad
│   └── 6/                  # Surprise
│
└── test/                   # Test dataset
    ├── 0/
    ├── 1/
    ├── 2/
    ├── 3/
    ├── 4/
    ├── 5/
    └── 6/
```

## 🔧 Configuration

### Model Settings

The application automatically looks for the model file in:
1. Parent directory: `../mod_my_model01.keras`
2. Current directory: `mod_my_model01.keras`

Edit `webapp.py` to modify:
- `IMG_SIZE`: Image preprocessing size (default: 224x224)

### UI Customization

Modify the CSS in `webapp.py` to customize:
- Colors and gradients
- Fonts and sizes
- Animations and transitions
- Layout and spacing

## 🐳 Docker Details

### Build Context

When building from the `webapp` directory, the Dockerfile uses the parent directory as context to access the model file.

### Environment Variables

- `STREAMLIT_SERVER_PORT`: Port for Streamlit server (default: 8501)
- `STREAMLIT_SERVER_ADDRESS`: Server address (default: 0.0.0.0)

## 📊 Model Information

- **Architecture**: MobileNetV2 (Transfer Learning)
- **Input Size**: 224x224 pixels
- **Classes**: 7 emotions
- **Framework**: TensorFlow/Keras

## 🛠️ Troubleshooting

### Model Not Found Error

If you see "Error loading model":
1. Ensure `mod_my_model01.keras` exists in the parent directory
2. Train the model using `face022.ipynb` if needed
3. Check file permissions
4. Verify the path in the error message

### Webcam Not Working

1. Ensure webcam is connected and not used by another application
2. Check browser permissions for camera access
3. Try different camera indices in the code (0, 1, 2)

### Docker Issues

1. **Port already in use**: Change port mapping `-p 8502:8501`
2. **Permission denied**: Run with `sudo` or add user to docker group
3. **Out of memory**: Increase Docker memory limit
4. **Model not found**: Ensure model file is mounted correctly in volumes

## 🚀 Deployment

### Cloud Platforms

#### Heroku
```bash
# Create Procfile in webapp directory
echo "web: streamlit run webapp.py --server.port=\$PORT --server.address=0.0.0.0" > Procfile

# Deploy
git push heroku main
```

#### AWS EC2
```bash
# SSH into instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Install Docker
sudo apt-get update
sudo apt-get install docker.io

# Run container
sudo docker run -d -p 8501:8501 emotion-app
```

#### Google Cloud Run
```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/PROJECT_ID/emotion-app

# Deploy
gcloud run deploy emotion-app --image gcr.io/PROJECT_ID/emotion-app --platform managed
```

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on GitHub.

## 🙏 Acknowledgments

- FER2013 dataset for training data
- TensorFlow team for the deep learning framework
- Streamlit team for the amazing web framework
- OpenCV for computer vision capabilities

---

**Made with ❤️ using Streamlit and TensorFlow**

