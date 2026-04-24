# # app.py - Flask Backend for Deepfake Detection
from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from werkzeug.utils import secure_filename
import uuid
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'deepfake-detector-secret-key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max

# Allowed extensions
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

# Load your trained model
try:
    model = keras.models.load_model("trained_deepfake_model/deepfake_detector.h5")
    print("✅ Deepfake detection model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_enhanced_features(video_path, max_frames=20):
    """Extract features using the same method as training"""
    print(f"🔍 Extracting features from: {os.path.basename(video_path)}")
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames-1, max_frames, dtype=int)
        
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (224, 224))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame / 255.0
                frames.append(frame)
        
        if len(frames) > 0:
            # Use multiple feature extractors (same as training)
            feature_extractors = [
                tf.keras.applications.EfficientNetB0(
                    include_top=False, weights='imagenet', pooling='avg'
                ),
                tf.keras.applications.MobileNetV2(
                    include_top=False, weights='imagenet', pooling='avg'
                )
            ]
            
            all_features = []
            frames_array = np.array(frames)
            
            for extractor in feature_extractors:
                extractor.trainable = False
                features = extractor.predict(frames_array, verbose=0)
                # Multiple pooling strategies
                mean_features = np.mean(features, axis=0)
                max_features = np.max(features, axis=0)
                std_features = np.std(features, axis=0)
                all_features.extend([mean_features, max_features, std_features])
            
            # Combine all features
            combined_features = np.concatenate(all_features)
            print(f"✅ Extracted {len(frames)} frames with {len(combined_features)} features")
            return combined_features
        else:
            return None
            
    except Exception as e:
        print(f"❌ Feature extraction error: {e}")
        return None
    finally:
        cap.release()

def predict_deepfake(video_path):
    """Predict if video is real or fake"""
    if model is None:
        return "MODEL_NOT_LOADED", 50.0, "Model not available"
    
    try:
        features = extract_enhanced_features(video_path)
        if features is None:
            return "ERROR", 0.0, "Could not extract features from video"
        
        # Reshape for prediction
        features = features.reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features, verbose=0)[0][0]
        confidence = float(prediction)
        
        # Determine result
        if prediction > 0.5:
            result = "FAKE"
            confidence_percent = confidence * 100
            explanation = "This video shows characteristics consistent with AI-generated content."
        else:
            result = "REAL" 
            confidence_percent = (1 - confidence) * 100
            explanation = "This video appears to be authentic with no significant deepfake indicators."
        
        return result, confidence_percent, explanation
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return "ERROR", 0.0, f"Analysis failed: {str(e)}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'error': 'No video selected'}), 400
    
    if file and allowed_file(file.filename):
        # Generate unique filename
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        try:
            file.save(filepath)
            
            # Analyze the video
            start_time = time.time()
            result, confidence, explanation = predict_deepfake(filepath)
            analysis_time = time.time() - start_time
            
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
            
            if result == "ERROR":
                return jsonify({'error': explanation}), 500
            elif result == "MODEL_NOT_LOADED":
                return jsonify({'error': 'Detection model is not available'}), 500
            else:
                return jsonify({
                    'result': result,
                    'confidence': round(confidence, 2),
                    'explanation': explanation,
                    'analysis_time': round(analysis_time, 2),
                    'status': 'success'
                })
                
        except Exception as e:
            # Clean up on error
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'Processing error: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type. Supported formats: MP4, AVI, MOV, MKV, WEBM'}), 400

@app.route('/health')
def health_check():
    """Health check endpoint"""
    model_status = "loaded" if model else "not loaded"
    return jsonify({
        'status': 'healthy',
        'model': model_status,
        'timestamp': time.time()
    })

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    print("🚀 Deepfake Detection Web App Starting...")
    print("📍 Access the app at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
# app.py - OPTIMIZED Flask Backend for Deepfake Detection
# app.py - OPTIMIZED Flask Backend for Deepfake Detection
