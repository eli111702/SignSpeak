import os
import io
import json
import base64
import threading
from collections import deque, defaultdict
from functools import wraps

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
from PIL import Image
import cv2
import mediapipe as mp
import tensorflow as tf

# ---------- CONFIG ----------
SEQUENCE_LENGTH = 30
FEATURE_SIZE = 63  # 21 landmarks * (x,y,z)
# UPDATED: Point to the 'model' subdirectory within the default 'models' directory
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# Map model_key -> filenames (model file, labels file)
ALLOWED_MODELS = {
    "alphabet_asl": {
        "model": "cnn_lstm_asl_model.keras",
        "labels": "ASL_labels_cnn_lstm.npy"
    },
    "digits": {
        "model": "cnn_lstm_num_model.keras",
        "labels": "numbers_labels_cnn_lstm.npy"
    },
    "alphabet_mysl": {
        "model": "cnn_lstm_MySL_model.keras",
        "labels": "MySL_labels_cnn_lstm.npy"
    },
    "basic_mysl": {
        "model": "mysl_SW_cnn_lstm_model.keras",
        "labels": "mysl_SW_labels.npy"
    }
}

# Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# MediaPipe Hands (single solver reused)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils 
# Using a context manager for the Hands object is best practice, but for global usage, instantiating it once is necessary.
mp_hands_solver = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# in-memory buffers: client_id -> deque([...kp arrays...])
buffers = defaultdict(lambda: deque(maxlen=SEQUENCE_LENGTH))
buffers_lock = threading.Lock()

# Model cache: model_key -> (tf_model, labels_list)
models_lock = threading.Lock()
loaded_models = {}

# ---------- UTIL: label loaders ----------
def load_labels_auto(path):
    """Loads labels from .npy, .txt, or .json files."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path, allow_pickle=True).tolist()
    elif ext == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported label format: {ext}")

# ---------- UTIL: image encoding ----------
def encode_bgr_image(bgr_image, quality=60):
    """Encodes a BGR image array to a base64 JPEG data URL."""
    # Encode BGR image to JPEG format in memory
    is_success, buffer = cv2.imencode(".jpg", bgr_image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not is_success:
        return None
    # Convert buffer to base64 string
    base64_encoded = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_encoded}"

# ---------- UTIL: image decoding and keypoint extraction (UPDATED) ----------
def decode_base64_image(data_url):
    """Decodes a base64 data URL into an OpenCV BGR image array."""
    if data_url.startswith("data:"):
        _, encoded = data_url.split(",", 1)
    else:
        encoded = data_url
    binary = base64.b64decode(encoded)
    # Open PIL Image, convert to RGB, then convert to BGR NumPy array for OpenCV compatibility
    image = Image.open(io.BytesIO(binary)).convert("RGB")
    arr = np.array(image)[:, :, ::-1].copy()
    return arr

def extract_hand_keypoints_from_bgr(bgr_image, draw_landmarks=False):
    """
    Processes BGR image to extract MediaPipe hand landmarks.
    
    Returns:
        tuple: (keypoints, processed_bgr_image, has_hand: bool)
    """
    # Create a copy for drawing if requested
    annotated_image = bgr_image.copy()
    
    img_rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    results = mp_hands_solver.process(img_rgb)
    
    keypoints = np.zeros(FEATURE_SIZE, dtype=np.float32)
    has_hand = False

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        kp_list = []
        for lm in hand_landmarks.landmark:
            # Extract x, y, z coordinates
            kp_list.extend([lm.x, lm.y, lm.z])
        
        # Fill the keypoints array and set has_hand flag
        if len(kp_list) == FEATURE_SIZE:
            keypoints = np.array(kp_list, dtype=np.float32)
            has_hand = True
            
        if draw_landmarks:
            # Draw the landmarks on the BGR image copy
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )
    
    # Always return the keypoints (either zeros or actual), the image, and the hand status.
    return keypoints, annotated_image, has_hand


# ---------- MODEL LOADER (Improved) ----------
def load_model_for_key(model_key):
    """Loads a model and its labels, caching the result."""
    if model_key not in ALLOWED_MODELS:
        print(f"[app] ERROR: Model key '{model_key}' is not defined in ALLOWED_MODELS.")
        return None, None
    
    with models_lock:
        if model_key in loaded_models:
            return loaded_models[model_key]
        
        info = ALLOWED_MODELS[model_key]
        model_path = os.path.join(MODELS_DIR, info["model"])
        labels_path = os.path.join(MODELS_DIR, info["labels"])

        print(f"\n[DEBUG] Attempting to load model key: '{model_key}'")
        print(f"[DEBUG] Model Path: {model_path}")
        print(f"[DEBUG] Labels Path: {labels_path}")

        # Check if the model_path exists
        if not os.path.exists(model_path):
            print(f"[app] ERROR: Missing model file: {model_path}")
            loaded_models[model_key] = (None, None)
            return None, None
        
        # Check if the labels_path exists
        if not os.path.exists(labels_path):
            print(f"[app] ERROR: Missing labels file: {labels_path}")
            loaded_models[model_key] = (None, None)
            return None, None
        
        try:
            # Try loading the model with compile=False, which often fixes compatibility issues
            tf_model = tf.keras.models.load_model(model_path, compile=False) 
        except Exception as e:
            print(f"[app] FATAL ERROR: Failed to load model {model_path}. Detailed error below:")
            print("-------------------------------------------------------------------------")
            print(e)
            print("-------------------------------------------------------------------------")
            loaded_models[model_key] = (None, None)
            return None, None
        
        try:
            labels = load_labels_auto(labels_path)
        except Exception as e:
            print(f"[app] FATAL ERROR: Failed to load labels {labels_path}. Error: {e}")
            loaded_models[model_key] = (None, None)
            return None, None
            
        loaded_models[model_key] = (tf_model, labels)
        print(f"[app] SUCCESS: Loaded model '{model_key}' -> {model_path} ({len(labels)} labels)")
        return tf_model, labels

# ---------- ROUTES ----------
@app.route("/")
def camera_permission():
    return render_template("camera_permission.html")

@app.route("/select")
def model_selection():
    return render_template("model_selection.html", models=ALLOWED_MODELS)

@app.route("/interpret/<model_key>")
def interpret_page(model_key):
    if model_key not in ALLOWED_MODELS:
        return "Unknown model", 404
    return render_template("interpreter.html", model_key=model_key, sequence_length=SEQUENCE_LENGTH)

@app.route("/predict", methods=["POST"])
def predict():
    req = request.get_json()
    if not req:
        return jsonify({"error": "no json"}), 400

    client_id = req.get("client_id")
    model_key = req.get("model_key")
    image_data = req.get("image")
    if not client_id or not model_key or not image_data:
        return jsonify({"error": "missing fields"}), 400

    # 1. Decode Image and Extract Keypoints (with landmark drawing)
    try:
        bgr = decode_base64_image(image_data)
    except Exception as e:
        return jsonify({"error": "bad image", "detail": str(e)}), 400

    kp, annotated_bgr, has_hand = extract_hand_keypoints_from_bgr(bgr, draw_landmarks=True)

    # 2. Encode the annotated frame back to base64 for the frontend
    annotated_image_data = encode_bgr_image(annotated_bgr)

    # 3. Handle No Hand Detected
    if not has_hand:
        # Clear the buffer if no hand is detected, effectively resetting the sequence
        with buffers_lock:
            buffers[client_id].clear()
        return jsonify({
            "status": "no_hand",
            "message": "No hand detected. Please position your hand clearly.",
            "annotated_image": annotated_image_data,
            "progress": 0,
            "needed": SEQUENCE_LENGTH
        })

    # 4. Buffer Sequence
    with buffers_lock:
        buffers[client_id].append(kp.copy())
        progress = len(buffers[client_id])

    # 5. Load Model (Cached)
    model, labels = load_model_for_key(model_key)
    if model is None or labels is None:
        # Check if model loading failed previously (it logs a detailed error on failure)
        return jsonify({"status": "error", "message": "Model or labels failed to load. Check server logs."}), 500

    # 6. Check Sequence Length
    if progress < SEQUENCE_LENGTH:
        return jsonify({
            "status": "collecting", 
            "progress": progress, 
            "needed": SEQUENCE_LENGTH,
            "annotated_image": annotated_image_data # Send annotated image regardless of prediction status
        })

    # 7. Predict
    with buffers_lock:
        seq = np.array(buffers[client_id], dtype=np.float32)
        # Sequence is complete, clear buffer to start next prediction immediately
        buffers[client_id].clear() 

    seq = np.expand_dims(seq, axis=0) # Add batch dimension

    try:
        preds = model.predict(seq, verbose=0)[0]
    except Exception as e:
        return jsonify({"status": "error", "message": f"model prediction error: {e}"}), 500

    idx = int(np.argmax(preds))
    label = str(labels[idx]) if idx < len(labels) else "unknown"
    confidence = float(preds[idx])

    return jsonify({
        "status": "predicted", 
        "label": label, 
        "confidence": confidence, 
        "progress": progress, # This will be SEQUENCE_LENGTH at time of prediction
        "annotated_image": annotated_image_data # Send annotated image along with prediction
    })

# NEW ROUTE: To immediately stop/clear the interpretation sequence
@app.route("/stop_interpretation", methods=["POST"])
def stop_interpretation():
    req = request.get_json()
    client_id = req.get("client_id")

    if not client_id:
        return jsonify({"error": "missing client_id"}), 400

    with buffers_lock:
        if client_id in buffers:
            buffers[client_id].clear()
            # Remove from dict if buffer is cleared
            if not buffers[client_id]:
                del buffers[client_id] 
    
    return jsonify({"status": "stopped", "message": "Interpretation sequence cleared."})

# ---------- start ----------
if __name__ == "__main__":
    # The fix ensures this directory structure is created if it doesn't exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("Starting app. Place your models in:", MODELS_DIR)
    app.run(host="0.0.0.0", port=5000, debug=True)