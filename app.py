import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable GPU probing (important)

import uuid
import numpy as np
import tensorflow as tf
import cv2
from flask import Flask, render_template, request, flash
from tensorflow.keras.preprocessing import image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime
from uuid import uuid4

# ------------------ APP SETUP ------------------
app = Flask(__name__)
app.config["SECRET_KEY"] = "DINESH@2006"

UPLOAD_FOLDER = "static/uploads"
import tempfile
import os

MODEL_URL = "https://huggingface.co/dinesh49/pneumonia-resnet50/resolve/main/pneumonia_resnet50.tflite"
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(BASE_DIR, "models", "pneumonia_resnet50.tflite")
MODEL_PATH = "/tmp/pneumonia_resnet50.tflite"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def ensure_model_ready():
    if not os.path.exists(MODEL_PATH):
        print("⬇️ Downloading TFLite model...")
        r = requests.get(MODEL_URL, stream=True)
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        print("✅ Model downloaded")
ensure_model_ready()


# ------------------ MODEL (LAZY LOAD) ------------------
# ------------------ TFLITE MODEL (LAZY LOAD) ------------------
interpreter = None
input_details = None
output_details = None

import requests

def download_model_if_needed():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print("Downloading TFLite model from Hugging Face...")
        r = requests.get(MODEL_URL, stream=True)
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        print("Model downloaded.")


def load_model_once():
    global interpreter, input_details, output_details
    if interpreter is None:
        print("🧠 Loading TFLite model...")
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("✅ Model loaded successfully")

    return interpreter




def startup_check():
    print("🔄 Checking model...")
    download_model_if_needed()

    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"📦 Model size: {size_mb:.2f} MB")

    if size_mb < 5:
        raise RuntimeError("❌ Model file corrupted or incomplete")

    print("✅ Model file ready")

# ------------------ IMAGE PREPROCESS ------------------
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img) / 255.0
    return np.expand_dims(img, axis=0)

# ------------------ X-RAY VALIDATION ------------------
def is_likely_xray(img_array):
    img = img_array[0]
    gray = np.mean(img, axis=-1)

    # 1️⃣ Color rejection
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    if np.mean(np.abs(r - g)) + np.mean(np.abs(g - b)) > 0.1:
        return False

    # 2️⃣ Contrast
    if np.std(gray) < 0.04:
        return False

    # 3️⃣ Aspect ratio
    h, w = gray.shape
    if not (0.6 <= w / h <= 1.4):
        return False

    # 4️⃣ White background
    if np.mean(gray > 0.9) > 0.35:
        return False

    # 5️⃣ Edge density
    edges = cv2.Canny((gray * 255).astype(np.uint8), 60, 140)
    if np.sum(edges > 0) / edges.size > 0.3:
        return False

    # 6️⃣ Texture smoothness (KEY FIX)
    lap = cv2.Laplacian((gray * 255).astype(np.uint8), cv2.CV_64F)
    if np.var(lap) > 1200:
        return False

    return True









# ------------------ PREDICTION ------------------
def predict_pneumonia(img_array):
    interpreter = load_model_once()

    img_array = img_array.astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    prob = float(output[0][0]) * 100

    if prob >= 75:
        return "PNEUMONIA", prob
    elif prob <= 30:
        return "NORMAL", prob
    else:
        return "UNCERTAIN", prob
    
    
def explain_prediction(label, confidence):
    if label == "PNEUMONIA":
        return [
            "Abnormal lung opacity patterns detected",
            "Reduced clarity in lung air regions",
            "Texture patterns similar to pneumonia cases",
            f"High confidence score of {confidence:.2f}%"
        ]
    else:
        return [
            "Clear lung fields detected",
            "Normal lung contrast distribution",
            "No abnormal opacity patterns found",
            f"Confidence score of {confidence:.2f}%"
        ]


def get_severity(confidence):
    if confidence >= 90:
        return "Severe"
    elif confidence >= 70:
        return "Moderate"
    else:
        return "Mild"


def get_reliability(confidence):
    if confidence >= 85:
        return "High reliability"
    elif confidence >= 60:
        return "Medium reliability"
    else:
        return "Low reliability"



# ------------------ GRAD-CAM ------------------
# def generate_gradcam(img_array):
#     model = load_model_once()
#     base_model = model.get_layer("resnet50")
#     last_conv = base_model.get_layer("conv5_block3_out")

#     grad_model = tf.keras.models.Model(
#         base_model.input,
#         [last_conv.output, base_model.output]
#     )

#     with tf.GradientTape() as tape:
#         conv_out, preds = grad_model(img_array)
#         loss = preds[:, 0]

#     grads = tape.gradient(loss, conv_out)
#     pooled = tf.reduce_mean(grads, axis=(0, 1, 2))

#     heatmap = tf.reduce_sum(conv_out[0] * pooled, axis=-1)
#     heatmap = np.maximum(heatmap, 0)
#     heatmap /= np.max(heatmap) + 1e-8

#     return heatmap

# def overlay_gradcam(img_path, heatmap, output_path):
#     img = cv2.resize(cv2.imread(img_path), (224, 224))
#     heatmap = cv2.resize(heatmap, (224, 224))
#     heatmap = np.uint8(255 * heatmap)
#     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#     overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
#     cv2.imwrite(output_path, overlay)

# ------------------ PDF ------------------
def generate_pdf(label, confidence, img_path, pdf_path):
    c = canvas.Canvas(pdf_path, pagesize=A4)
    w, h = A4
    y = h - 40

    c.setFont("Helvetica-Bold", 18)
    c.drawString(40, y, "PneumoScan AI – Chest X-ray Analysis Report")

    y -= 25
    c.setFont("Helvetica", 10)
    c.drawString(40, y, f"Date: {datetime.now()}")
    c.drawString(300, y, "Model: ResNet50 (TFLite)")

    y -= 25
    c.line(40, y, w - 40, y)

    y -= 25
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, f"Diagnosis: {label}")
    c.drawString(300, y, f"Confidence: {confidence:.2f}%")

    y -= 40
    c.drawImage(img_path, 40, y - 150, 200, 150)

    y -= 180
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(40, y, "Educational use only. Not a medical diagnosis.")

    c.save()


# ------------------ ROUTE ------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_path = None
    pdf_path = None

    explanation = None      # ✅ ADD
    severity = None         # ✅ ADD
    reliability = None      # ✅ ADD

    prediction = confidence = image_path = pdf_path = None

    if request.method == "POST":
        file = request.files.get("image")

        if not file:
            flash("No image selected", "error")
            return render_template("index.html")

        uid = uuid.uuid4().hex
        image_path = f"{UPLOAD_FOLDER}/{uid}.jpg"
        pdf_path = f"{UPLOAD_FOLDER}/{uid}_report.pdf"

        file.save(image_path)
        img_array = preprocess_image(image_path)

        # X-ray validation (ONLY ONCE)
        if not is_likely_xray(img_array):
            os.remove(image_path)
            flash("Invalid image. Please upload a valid chest X-ray image.", "warning")
            return render_template("index.html")

        # Prediction (ONLY ONCE)
        prediction, confidence = predict_pneumonia(img_array)
        explanation = explain_prediction(prediction, confidence)
        severity = get_severity(confidence)
        reliability = get_reliability(confidence)


        if prediction == "UNCERTAIN":
            os.remove(image_path)
            flash("Low confidence prediction. Try a clearer X-ray.", "warning")
            return render_template("index.html")

        generate_pdf(prediction, confidence, image_path, pdf_path)

    return render_template(
    "index.html",
    prediction=prediction,
    confidence=confidence,
    explanation=explanation,
    severity=severity,
    reliability=reliability,
    image_path=image_path,
    pdf_path=pdf_path,
    uuid4=uuid4
)


# ------------------ RUN ------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)




# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable GPU probing (important)

# import uuid
# import numpy as np
# import tensorflow as tf
# import cv2
# from flask import Flask, render_template, request, flash
# from tensorflow.keras.preprocessing import image
# from reportlab.lib.pagesizes import A4
# from reportlab.pdfgen import canvas
# from datetime import datetime
# from uuid import uuid4

# # ------------------ APP SETUP ------------------
# app = Flask(__name__)
# app.config["SECRET_KEY"] = "DINESH@2006"

# UPLOAD_FOLDER = "static/uploads"
# MODEL_PATH = os.path.join("models", "pneumonia_resnet50.tflite")

# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # ------------------ MODEL (LAZY LOAD) ------------------
# # ------------------ TFLITE MODEL (LAZY LOAD) ------------------
# interpreter = None
# input_details = None
# output_details = None

# def load_model_once():
#     global interpreter, input_details, output_details
#     if interpreter is None:
#         print("Loading TFLite model...")
#         interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
#         interpreter.allocate_tensors()
#         input_details = interpreter.get_input_details()
#         output_details = interpreter.get_output_details()
#         print("TFLite model loaded.")
#     return interpreter


# # ------------------ IMAGE PREPROCESS ------------------
# def preprocess_image(img_path):
#     img = image.load_img(img_path, target_size=(224, 224))
#     img = image.img_to_array(img) / 255.0
#     return np.expand_dims(img, axis=0)

# # ------------------ X-RAY VALIDATION ------------------
# def is_likely_xray(img_array):
#     img = img_array[0]
#     gray = np.mean(img, axis=-1)

#     # 1️⃣ Color rejection
#     r, g, b = img[..., 0], img[..., 1], img[..., 2]
#     if np.mean(np.abs(r - g)) + np.mean(np.abs(g - b)) > 0.1:
#         return False

#     # 2️⃣ Contrast
#     if np.std(gray) < 0.04:
#         return False

#     # 3️⃣ Aspect ratio
#     h, w = gray.shape
#     if not (0.6 <= w / h <= 1.4):
#         return False

#     # 4️⃣ White background
#     if np.mean(gray > 0.9) > 0.35:
#         return False

#     # 5️⃣ Edge density
#     edges = cv2.Canny((gray * 255).astype(np.uint8), 60, 140)
#     if np.sum(edges > 0) / edges.size > 0.3:
#         return False

#     # 6️⃣ Texture smoothness (KEY FIX)
#     lap = cv2.Laplacian((gray * 255).astype(np.uint8), cv2.CV_64F)
#     if np.var(lap) > 1200:
#         return False

#     return True









# # ------------------ PREDICTION ------------------
# def predict_pneumonia(img_array):
#     interpreter = load_model_once()

#     img_array = img_array.astype(np.float32)

#     interpreter.set_tensor(input_details[0]['index'], img_array)
#     interpreter.invoke()

#     output = interpreter.get_tensor(output_details[0]['index'])
#     prob = float(output[0][0]) * 100

#     if prob >= 75:
#         return "PNEUMONIA", prob
#     elif prob <= 30:
#         return "NORMAL", prob
#     else:
#         return "UNCERTAIN", prob
    
    
# def explain_prediction(label, confidence):
#     if label == "PNEUMONIA":
#         return [
#             "Abnormal lung opacity patterns detected",
#             "Reduced clarity in lung air regions",
#             "Texture patterns similar to pneumonia cases",
#             f"High confidence score of {confidence:.2f}%"
#         ]
#     else:
#         return [
#             "Clear lung fields detected",
#             "Normal lung contrast distribution",
#             "No abnormal opacity patterns found",
#             f"Confidence score of {confidence:.2f}%"
#         ]


# def get_severity(confidence):
#     if confidence >= 90:
#         return "Severe"
#     elif confidence >= 70:
#         return "Moderate"
#     else:
#         return "Mild"


# def get_reliability(confidence):
#     if confidence >= 85:
#         return "High reliability"
#     elif confidence >= 60:
#         return "Medium reliability"
#     else:
#         return "Low reliability"



# # ------------------ GRAD-CAM ------------------
# # def generate_gradcam(img_array):
# #     model = load_model_once()
# #     base_model = model.get_layer("resnet50")
# #     last_conv = base_model.get_layer("conv5_block3_out")

# #     grad_model = tf.keras.models.Model(
# #         base_model.input,
# #         [last_conv.output, base_model.output]
# #     )

# #     with tf.GradientTape() as tape:
# #         conv_out, preds = grad_model(img_array)
# #         loss = preds[:, 0]

# #     grads = tape.gradient(loss, conv_out)
# #     pooled = tf.reduce_mean(grads, axis=(0, 1, 2))

# #     heatmap = tf.reduce_sum(conv_out[0] * pooled, axis=-1)
# #     heatmap = np.maximum(heatmap, 0)
# #     heatmap /= np.max(heatmap) + 1e-8

# #     return heatmap

# # def overlay_gradcam(img_path, heatmap, output_path):
# #     img = cv2.resize(cv2.imread(img_path), (224, 224))
# #     heatmap = cv2.resize(heatmap, (224, 224))
# #     heatmap = np.uint8(255 * heatmap)
# #     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
# #     overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
# #     cv2.imwrite(output_path, overlay)

# # ------------------ PDF ------------------
# def generate_pdf(label, confidence, img_path, pdf_path):
#     c = canvas.Canvas(pdf_path, pagesize=A4)
#     w, h = A4
#     y = h - 40

#     c.setFont("Helvetica-Bold", 18)
#     c.drawString(40, y, "PneumoScan AI – Chest X-ray Analysis Report")

#     y -= 25
#     c.setFont("Helvetica", 10)
#     c.drawString(40, y, f"Date: {datetime.now()}")
#     c.drawString(300, y, "Model: ResNet50 (TFLite)")

#     y -= 25
#     c.line(40, y, w - 40, y)

#     y -= 25
#     c.setFont("Helvetica-Bold", 14)
#     c.drawString(40, y, f"Diagnosis: {label}")
#     c.drawString(300, y, f"Confidence: {confidence:.2f}%")

#     y -= 40
#     c.drawImage(img_path, 40, y - 150, 200, 150)

#     y -= 180
#     c.setFont("Helvetica-Oblique", 9)
#     c.drawString(40, y, "Educational use only. Not a medical diagnosis.")

#     c.save()


# # ------------------ ROUTE ------------------
# @app.route("/", methods=["GET", "POST"])
# def index():
#     prediction = None
#     confidence = None
#     image_path = None
#     pdf_path = None

#     explanation = None      # ✅ ADD
#     severity = None         # ✅ ADD
#     reliability = None      # ✅ ADD

#     prediction = confidence = image_path = pdf_path = None

#     if request.method == "POST":
#         file = request.files.get("image")

#         if not file:
#             flash("No image selected", "error")
#             return render_template("index.html")

#         uid = uuid.uuid4().hex
#         image_path = f"{UPLOAD_FOLDER}/{uid}.jpg"
#         pdf_path = f"{UPLOAD_FOLDER}/{uid}_report.pdf"

#         file.save(image_path)
#         img_array = preprocess_image(image_path)

#         # X-ray validation (ONLY ONCE)
#         if not is_likely_xray(img_array):
#             os.remove(image_path)
#             flash("Invalid image. Please upload a valid chest X-ray image.", "error")
#             return render_template("index.html")

#         # Prediction (ONLY ONCE)
#         prediction, confidence = predict_pneumonia(img_array)
#         explanation = explain_prediction(prediction, confidence)
#         severity = get_severity(confidence)
#         reliability = get_reliability(confidence)


#         if prediction == "UNCERTAIN":
#             os.remove(image_path)
#             flash("Low confidence prediction. Try a clearer X-ray.", "warning")
#             return render_template("index.html")

#         generate_pdf(prediction, confidence, image_path, pdf_path)

#     return render_template(
#     "index.html",
#     prediction=prediction,
#     confidence=confidence,
#     explanation=explanation,
#     severity=severity,
#     reliability=reliability,
#     image_path=image_path,
#     pdf_path=pdf_path,
#     uuid4=uuid4
# )


# # ------------------ RUN ------------------
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)
