import os
import random
import uuid
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SERVICE_ACCOUNT_PATH = os.path.join(BASE_DIR, "serviceAccountKey.json")

if not firebase_admin._apps:
    cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
    firebase_admin.initialize_app(cred)

db = firestore.client()


def generate_case():
    """
    Generate one synthetic case
    """

    # randomly choose real label
    ground_truth = random.choice(["Fracture", "Normal"])

    # simulate model prediction
    if ground_truth == "Fracture":

        confidence = random.uniform(0.6, 0.95)

        prediction = (
            "Fracture"
            if random.random() < 0.85
            else "Normal"
        )

    else:

        confidence = random.uniform(0.05, 0.4)

        prediction = (
            "Normal"
            if random.random() < 0.85
            else "Fracture"
        )

    case = {
        "prediction": prediction,
        "confidence": confidence,
        "fractureProbability": confidence,
        "riskLevel": (
            "High" if confidence >= 0.8
            else "Moderate" if confidence >= 0.5
            else "Low"
        ),
        "boxes": [],
        "originalImageBase64": "",
        "annotatedImageBase64": "",
        "gradCamBase64": "",
        "modelName": "EfficientNet-B3 + YOLOv8",
        "summary": "Synthetic evaluation case",
        "recommendation": "Test dataset entry",
        "groundTruth": ground_truth,
        "createdAt": firestore.SERVER_TIMESTAMP,
    }

    return case


def seed_cases(count=100):

    print(f"\nGenerating {count} test cases...\n")

    for i in range(count):

        case = generate_case()

        doc_id = str(uuid.uuid4())

        db.collection("cases").document(doc_id).set(case)

        if i % 10 == 0:
            print(f"{i} cases inserted")

    print("\nDone.")


if __name__ == "__main__":

    print("\nMEDORA Dataset Seeder\n")

    count = input("How many cases to generate? (default 100): ")

    if count.strip() == "":
        count = 100
    else:
        count = int(count)

    seed_cases(count)