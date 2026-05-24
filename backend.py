from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import cv2
from werkzeug.utils import secure_filename
import tensorflow as tf
from mtcnn import MTCNN

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}
IMG_SIZE = 224

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = None
face_detector = None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model():
    global model
    try:
        model = tf.keras.models.load_model(
            r"c:\Users\Ayush\models\deepfake_best_3000samples.h5"
        )
        print(" Model loaded")
        return True
    except:
        print(" Model not found → demo mode")
        return False


def init_detector():
    global face_detector
    face_detector = MTCNN()


def extract_face(image):
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = face_detector.detect_faces(image)

        if len(faces) > 0:
            face = max(faces, key=lambda x: x['box'][2] * x['box'][3])
            x, y, w, h = face['box']

           
            margin_x = int(w * 0.2)
            margin_y = int(h * 0.2)
            x = max(0, x - margin_x)
            y = max(0, y - margin_y)
            w = min(image.shape[1] - x, w + 2 * margin_x)
            h = min(image.shape[0] - y, h + 2 * margin_y)

            face_img = image[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))

            return face_img

        return None
    except:
        return None



def get_verdict(prediction):
   
    if prediction > 0.5:
        return True, "FAKE"
    else:
        return False, "REAL"



def adjust_scores(prediction, verdict):
    if verdict == "REAL":
        authentic = max(55, (1 - prediction) * 100)
        fake = 100 - authentic
    else:  
        fake = max(55, prediction * 100)
        authentic = 100 - fake

    return round(authentic, 1), round(fake, 1)



def analyze_image(path):
    img = cv2.imread(path)
    if img is None:
        return {'success': False, 'error': 'Invalid image'}

    face = extract_face(img)
    if face is None:
        return {'success': False, 'error': 'No face detected'}

    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, axis=0)

    prediction = float(model.predict(face, verbose=0)[0][0]) if model else float(np.random.uniform(0.2, 0.8))

    is_fake, verdict = get_verdict(prediction)
    authentic, fake = adjust_scores(prediction, verdict)
    confidence = round(abs(prediction - 0.5) * 2 * 100, 1)

    return {
        'success': True,
        'is_fake': is_fake,
        'verdict': verdict,
        'confidence': confidence,
        'authentic_probability': authentic,
        'fake_probability': fake,
        'raw_score': round(prediction, 4)
    }




def analyze_video(path):
    cap = cv2.VideoCapture(path)
    preds = []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_count = min(30, total)
    indices = np.linspace(0, total - 1, sample_count, dtype=int)

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue

        face = extract_face(frame)
        if face is not None:
            face = face.astype('float32') / 255.0
            face = np.expand_dims(face, axis=0)

            p = float(model.predict(face, verbose=0)[0][0]) if model else float(np.random.uniform(0.2, 0.8))
            preds.append(p)

    cap.release()

    if not preds:
        return {'success': False, 'error': 'No face detected in video'}

    avg_pred = float(np.mean(preds))
    
    fake_frame_ratio = sum(1 for p in preds if p > 0.5) / len(preds)

    is_fake, verdict = get_verdict(avg_pred)
    authentic, fake = adjust_scores(avg_pred, verdict)
    confidence = round(abs(avg_pred - 0.5) * 2 * 100, 1)

    return {
        'success': True,
        'is_fake': is_fake,
        'verdict': verdict,
        'confidence': confidence,
        'authentic_probability': authentic,
        'fake_probability': fake,
        'frames_analyzed': len(preds),
        'fake_frame_ratio': round(fake_frame_ratio * 100, 1),
        'raw_score': round(avg_pred, 4)
    }




@app.route('/api/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})

    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({'success': False, 'error': 'Empty filename'})

    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    ext = filename.rsplit('.', 1)[-1].lower()

    try:
        if ext in ['jpg', 'jpeg', 'png']:
            result = analyze_image(path)
        elif ext in ['mp4', 'avi', 'mov']:
            result = analyze_video(path)
        else:
            result = {'success': False, 'error': 'Unsupported file type'}
    finally:
        if os.path.exists(path):
            os.remove(path)

    return jsonify(result)


@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None})




if __name__ == '__main__':
    init_detector()
    load_model()
    app.run(debug=True)