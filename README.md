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

## PROJECT INTERFACE



### Home Page

![Home Page](project%20screenshots/Screenshot%202026-04-23%20131135.png)

### Image upload

![Dark Mode](project%20screenshots/Screenshot%202026-05-24%20170108.png)

### Video Upload

![Image Upload](project%20screenshots/Screenshot%202026-05-24%20170149.png)

### Analysis in Progress

![Video Upload](project%20screenshots/Screenshot%202026-05-24%20170229.png)

###  Fake Prediction

![Analysis](project%20screenshots/Screenshot%202026-05-24%20170343.png)

### Real Prediction

![Real Result](project%20screenshots/Screenshot%202026-05-24%20170416.png)

### Model Info Tab

![Fake Result](project%20screenshots/Screenshot%202026-05-24%20170453.png)

### Analytics Tab

![Fake Result](project%20screenshots/Screenshot%202026-05-24%20170512.png)

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

## Training the Model

### Training Script

The complete training code is provided in:

```text
training_script.txt
```

The file contains the original Python source code used for model training.

Because the training script is large, it has been uploaded in `.txt` format.

### How to Run

1. Download the repository.

2. Rename:

```text
training_script.txt
```

to:

```text
train_model.py
```

3. Open JupyterLab:

```bash
jupyter lab
```

or

```bash
jupyter notebook
```

4. Configure dataset locations:

```python
DATASET_PATHS = {
    'dfdc': 'path_to_dfdc',
    'faceforensics': 'path_to_faceforensics',
    'celebdf': 'path_to_celebdf'
}
```

5. Execute the training script.

### Recommended Environment

- Python 3.10+
- TensorFlow 2.15
- CUDA 11.x
- GPU (4GB+ VRAM)
- JupyterLab

### Generated Outputs

Training produces:

```text
deepfake_best_3000samples.h5
confusion_matrix.png
roc_curve.png
training_history.png
classification_report.txt
```

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
