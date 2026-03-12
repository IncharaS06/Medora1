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

app = FastAPI(title="MEDORA Backend", version="2.1.0")

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
    print("EfficientNet weights loaded successfully.")
else:
    print("WARNING: grazped_finetuned_best.pth not found.")

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


def clean_label(value: str | None) -> str | None:
    if value is None:
        return None

    label = str(value).strip()
    if not label:
        return None

    normalized = label.lower()
    if normalized in {"fracture", "positive", "1"}:
        return "Fracture"
    if normalized in {"normal", "negative", "0"}:
        return "Normal"

    return None


def image_to_base64(
    image_path: str,
    size: tuple[int, int] | None = None,
    quality: int = 80
) -> str:
    img = cv2.imread(image_path)
    if img is None:
        raise Exception(f"Could not read image: {image_path}")

    if size is not None:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    success, buffer = cv2.imencode(
        ".jpg",
        img,
        [int(cv2.IMWRITE_JPEG_QUALITY), quality],
    )

    if not success:
        raise Exception("Image compression failed")

    return base64.b64encode(buffer).decode("utf-8")


def predict_fracture_probability(image_path: str) -> float:
    image = Image.open(image_path).convert("RGB")
    input_tensor = eff_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = efficientnet_model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]

    # class index 1 = fracture
    return float(probs[1].item())


def serialize_firestore_value(value):
    if value is None:
        return None

    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            return str(value)

    return value


def serialize_firestore_doc(doc) -> dict:
    data = doc.to_dict() or {}
    data["id"] = doc.id

    if "createdAt" in data:
        data["createdAt"] = serialize_firestore_value(data.get("createdAt"))

    return data


@app.get("/")
def root():
    return {"message": "MEDORA backend is running"}


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    groundTruth: str | None = Form(default=None),
    patientName: str | None = Form(default=None),
    userId: str | None = Form(default=None),
    userEmail: str | None = Form(default=None),
):
    temp_filename = os.path.join(BASE_DIR, f"temp_{uuid.uuid4()}.png")
    annotated_file = None

    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        original_base64 = image_to_base64(
            temp_filename,
            size=(512, 512),
            quality=80
        )

        fracture_probability = predict_fracture_probability(temp_filename)

        results = yolo_model(temp_filename, conf=0.1)
        r = results[0]

        boxes = []
        yolo_confidence = 0.0

        if r.boxes is not None and len(r.boxes) > 0:
            yolo_confidence = float(r.boxes.conf.max().item())
            for box in r.boxes.xyxy:
                x1, y1, x2, y2 = box.tolist()
                boxes.append({
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                })

        prediction = "Fracture" if (len(boxes) > 0 or fracture_probability >= 0.5) else "Normal"
        confidence = max(float(fracture_probability), float(yolo_confidence))

        annotated = r.plot()
        annotated_file = os.path.join(BASE_DIR, f"annotated_{uuid.uuid4()}.jpg")
        cv2.imwrite(annotated_file, annotated)

        annotated_base64 = image_to_base64(
            annotated_file,
            size=(512, 512),
            quality=80
        )

        gradcam_base64 = generate_gradcam(
            efficientnet_model,
            temp_filename,
            target_class=1
        )

        risk_level = (
            "High" if confidence >= 0.8
            else "Moderate" if confidence >= 0.5
            else "Low"
        )

        summary = (
            "Suspicious fracture-related region detected in the wrist radiograph."
            if prediction == "Fracture"
            else "No strong fracture-related localization detected by the model."
        )

        recommendation = (
            "Clinical review recommended. Correlate with radiologist interpretation."
            if prediction == "Fracture"
            else "Model suggests a normal case, but clinical review is still advised."
        )

        normalized_ground_truth = clean_label(groundTruth)
        response_created_at = datetime.now(timezone.utc).isoformat()

        firestore_payload = {
            "patientName": patientName or "Unknown",
            "prediction": prediction,
            "confidence": confidence,
            "fractureProbability": fracture_probability,
            "yoloConfidence": yolo_confidence,
            "riskLevel": risk_level,
            "boxes": boxes,
            "originalImageBase64": original_base64,
            "annotatedImageBase64": annotated_base64,
            "gradCamBase64": gradcam_base64,
            "modelName": "EfficientNet-B3 + YOLOv8",
            "summary": summary,
            "recommendation": recommendation,
            "createdAt": firestore.SERVER_TIMESTAMP,
        }

        if userId:
            firestore_payload["userId"] = userId

        if userEmail:
            firestore_payload["userEmail"] = userEmail

        if normalized_ground_truth is not None:
            firestore_payload["groundTruth"] = normalized_ground_truth

        doc_ref = db.collection("cases").document()
        doc_ref.set(firestore_payload)

        response_payload = {
            "id": doc_ref.id,
            "patientName": patientName or "Unknown",
            "prediction": prediction,
            "confidence": confidence,
            "fractureProbability": fracture_probability,
            "yoloConfidence": yolo_confidence,
            "riskLevel": risk_level,
            "boxes": boxes,
            "originalImageBase64": original_base64,
            "annotatedImageBase64": annotated_base64,
            "gradCamBase64": gradcam_base64,
            "modelName": "EfficientNet-B3 + YOLOv8",
            "summary": summary,
            "recommendation": recommendation,
            "createdAt": response_created_at,
        }

        if userId:
            response_payload["userId"] = userId

        if userEmail:
            response_payload["userEmail"] = userEmail

        if normalized_ground_truth is not None:
            response_payload["groundTruth"] = normalized_ground_truth

        return response_payload

    except Exception as e:
        print("ANALYZE ERROR:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        if annotated_file and os.path.exists(annotated_file):
            os.remove(annotated_file)


@app.get("/cases")
def get_cases():
    try:
        docs = db.collection("cases").order_by(
            "createdAt",
            direction=firestore.Query.DESCENDING
        ).stream()

        return [serialize_firestore_doc(doc) for doc in docs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cases/{case_id}")
def get_case(case_id: str):
    try:
        doc = db.collection("cases").document(case_id).get()
        if not doc.exists:
            raise HTTPException(status_code=404, detail="Case not found")
        return serialize_firestore_doc(doc)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/cases/{case_id}/label")
def update_case_label(case_id: str, groundTruth: str = Form(...)):
    try:
        gt = clean_label(groundTruth)

        if gt not in {"Fracture", "Normal"}:
            raise HTTPException(
                status_code=400,
                detail="groundTruth must be Fracture or Normal"
            )

        ref = db.collection("cases").document(case_id)
        snap = ref.get()

        if not snap.exists:
            raise HTTPException(status_code=404, detail="Case not found")

        ref.update({"groundTruth": gt})
        updated = ref.get()
        return serialize_firestore_doc(updated)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/evaluation")
def get_evaluation():
    try:
        docs = db.collection("cases").stream()

        rows = []
        for doc in docs:
            data = doc.to_dict() or {}
            rows.append({
                "groundTruth": data.get("groundTruth") or data.get("actualLabel") or data.get("trueLabel"),
                "prediction": data.get("prediction"),
                "score": float(data.get("fractureProbability", data.get("confidence", 0.0)) or 0.0),
            })

        evaluation = compute_binary_metrics(rows)
        return evaluation

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))