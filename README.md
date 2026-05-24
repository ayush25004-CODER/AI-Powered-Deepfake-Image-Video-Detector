# AI-Powered-Deepfake-Image-Video-Detector
## Overview

AI Powered Deepfake Detector is a web-based application that identifies manipulated images and videos using Deep Learning.

The system combines:

- EfficientNet-B4
- MTCNN Face Detection
- TensorFlow/Keras
- OpenCV
- Flask Backend
- HTML/CSS/JavaScript Frontend

The application allows users to upload an image or video and instantly receive a prediction indicating whether the media is REAL or FAKE.

---

## Features

### Image Detection

Detects fake facial manipulations in:

- JPG
- JPEG
- PNG

### Video Detection

Detects deepfake videos by:

- Extracting multiple frames
- Detecting faces
- Averaging frame predictions

Supports:

- MP4
- AVI
- MOV

### Face Extraction

Uses MTCNN for:

- Face localization
- Face cropping
- Face preprocessing

### Confidence Scoring

Displays:

- Authenticity Score
- Manipulation Score
- Confidence Percentage

### Modern UI

Includes:

- Dark Mode
- Drag & Drop Upload
- Responsive Design
- Animated Dashboard

---

## Screenshots

### Home Page

![Home Page](screenshots/homepage.png)

### Dark Mode

![Dark Mode](screenshots/darkmode.png)

### Image Upload

![Image Upload](screenshots/upload-image.png)

### Video Upload

![Video Upload](screenshots/upload-video.png)

### Analysis in Progress

![Analysis](screenshots/analyzing.png)

### Real Prediction

![Real Result](screenshots/result-real.png)

### Fake Prediction

![Fake Result](screenshots/result-fake.png)

### Model Information

![Model Info](screenshots/model-info.png)

### Analytics Dashboard

![Analytics](screenshots/analytics-page.png)

## Technologies Used

### Frontend

HTML5

Tailwind CSS

JavaScript

### Backend

Flask

Flask-CORS

TensorFlow

NumPy

OpenCV

MTCNN

### Machine Learning

EfficientNet-B4

Transfer Learning

Binary Classification

---

## Datasets Used

### DFDC

DeepFake Detection Challenge Dataset

### FaceForensics++

Large-scale manipulated facial video dataset

### Celeb-DF

High quality deepfake video dataset

---

## Project Architecture

User Upload

↓

Frontend Interface

↓

Flask API

↓

Face Detection (MTCNN)

↓

EfficientNet-B4 Model

↓

Prediction

↓

Result Dashboard

---

## Installation

### Clone Repository

git clone https://github.com/YOUR_USERNAME/AI-Powered-Deepfake-Image-Video-Detector.git

cd AI-Powered-Deepfake-Image-Video-Detector

---

### Create Virtual Environment

python -m venv venv

Windows

venv\Scripts\activate

Linux

source venv/bin/activate

---

### Install Dependencies

pip install -r requirements.txt

---

### Run Backend

python backend/app.py

Backend starts at:

http://localhost:5000

---

### Open Frontend

Open:

frontend/index.html

in browser

---

## API Endpoints

### Analyze Media

POST

/api/analyze

Supported:

Images

Videos

---

### Health Check

GET

/api/health

Returns:

{
  "status":"ok",
  "model_loaded":true
}

---

## Model Details

Base Network:

EfficientNet-B4

Input Size:

224 × 224

Output:

Binary Classification

0 → REAL

1 → FAKE

Training Samples:

9000+

Datasets:

DFDC

FaceForensics++

Celeb-DF

---

## Future Improvements

Grad-CAM Explainability

Live Webcam Detection

Batch Processing

Mobile Application

Cloud Deployment

Docker Support

---

## Authors

Ayush Upreti

Vedic Rawat

Atul Kumar

Graphic Era Hill University

Department of Computer Science & Engineering

2026

---

## License

MIT License
