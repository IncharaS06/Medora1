from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import shutil
import uuid
import cv2
import os
import base64
import firebase_admin
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from datetime import datetime, timezone

from firebase_admin import credentials, firestore
from gradcam import generate_gradcam
from metrics_utils import compute_binary_metrics

app = FastAPI(title="MEDORA Backend", version="2.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "model", "best.pt")
EFF_MODEL_PATH = os.path.join(BASE_DIR, "model", "grazped_finetuned_best.pth")
SERVICE_ACCOUNT_PATH = os.path.join(BASE_DIR, "serviceAccountKey.json")

if not firebase_admin._apps:
    cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
    firebase_admin.initialize_app(cred)

db = firestore.client()
yolo_model = YOLO(YOLO_MODEL_PATH)


# ---------------- EfficientNet ----------------

class EffB3(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = models.efficientnet_b3(weights=None)
        in_f = self.m.classifier[1].in_features
        self.m.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_f, 2)
        )

    def forward(self, x):
        return self.m(x)


efficientnet_model = EffB3()

if os.path.exists(EFF_MODEL_PATH):
    state = torch.load(EFF_MODEL_PATH, map_location="cpu")
    efficientnet_model.load_state_dict(state, strict=False)
    print("EfficientNet weights loaded.")
else:
    print("WARNING: model not found")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
efficientnet_model.to(DEVICE)
efficientnet_model.eval()

eff_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ---------------- VALIDATION ----------------

def validate_wrist_xray(file: UploadFile, temp_path: str):

    # type check
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Only image allowed"
        )

    # filename check
    name = (file.filename or "").lower()

    if "wrist" not in name and "xray" not in name:
        raise HTTPException(
            status_code=400,
            detail="Upload only wrist xray"
        )

    # grayscale check
    img = cv2.imread(temp_path)

    if img is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image"
        )

    b, g, r = cv2.split(img)

    diff1 = cv2.absdiff(b, g).mean()
    diff2 = cv2.absdiff(g, r).mean()
    diff3 = cv2.absdiff(b, r).mean()

    if diff1 > 15 or diff2 > 15 or diff3 > 15:
        raise HTTPException(
            status_code=400,
            detail="Only X-ray allowed"
        )


# ---------------- UTILS ----------------

def image_to_base64(path, size=None, quality=80):
    img = cv2.imread(path)

    if size:
        img = cv2.resize(img, size)

    _, buf = cv2.imencode(".jpg", img,
        [int(cv2.IMWRITE_JPEG_QUALITY), quality])

    return base64.b64encode(buf).decode()


def predict_fracture_probability(path):

    image = Image.open(path).convert("RGB")
    x = eff_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = efficientnet_model(x)
        probs = torch.softmax(logits, dim=1)[0]

    return float(probs[1])


# ---------------- ROOT ----------------

@app.get("/")
def root():
    return {"ok": True}


# ---------------- ANALYZE ----------------

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    groundTruth: str | None = Form(None),
    patientName: str | None = Form(None),
    userId: str | None = Form(None),
    userEmail: str | None = Form(None),
):

    temp = os.path.join(
        BASE_DIR,
        f"temp_{uuid.uuid4()}.png"
    )

    annotated_file = None

    try:

        with open(temp, "wb") as buf:
            shutil.copyfileobj(file.file, buf)

        # ✅ VALIDATE
        validate_wrist_xray(file, temp)

        original_base64 = image_to_base64(temp, (512,512))

        fracture_prob = predict_fracture_probability(temp)

        results = yolo_model(temp, conf=0.1)
        r = results[0]

        boxes = []
        yolo_conf = 0

        if r.boxes:
            yolo_conf = float(
                r.boxes.conf.max().item()
            )

            for box in r.boxes.xyxy:
                x1,y1,x2,y2 = box.tolist()
                boxes.append({
                    "x1":x1,
                    "y1":y1,
                    "x2":x2,
                    "y2":y2,
                })

        prediction = (
            "Fracture"
            if boxes or fracture_prob>=0.5
            else "Normal"
        )

        confidence = max(
            fracture_prob,
            yolo_conf
        )

        annotated = r.plot()

        annotated_file = os.path.join(
            BASE_DIR,
            f"ann_{uuid.uuid4()}.jpg"
        )

        cv2.imwrite(annotated_file, annotated)

        annotated_base64 = image_to_base64(
            annotated_file,
            (512,512)
        )

        gradcam_base64 = generate_gradcam(
            efficientnet_model,
            temp,
            1
        )

        risk = (
            "High" if confidence>=0.8
            else "Moderate" if confidence>=0.5
            else "Low"
        )

        doc = db.collection("cases").document()

        doc.set({

            "prediction": prediction,
            "confidence": confidence,
            "riskLevel": risk,
            "boxes": boxes,
            "originalImageBase64": original_base64,
            "annotatedImageBase64": annotated_base64,
            "gradCamBase64": gradcam_base64,
            "createdAt": firestore.SERVER_TIMESTAMP

        })

        return {
            "id": doc.id,
            "prediction": prediction,
            "confidence": confidence,
            "riskLevel": risk,
            "boxes": boxes,
            "originalImageBase64": original_base64,
            "annotatedImageBase64": annotated_base64,
            "gradCamBase64": gradcam_base64,
        }

    except Exception as e:

        print("ERROR", e)

        raise HTTPException(
            500,
            str(e)
        )

    finally:

        if os.path.exists(temp):
            os.remove(temp)

        if annotated_file and os.path.exists(annotated_file):
            os.remove(annotated_file)
