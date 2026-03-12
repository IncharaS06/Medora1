import os
import firebase_admin
from firebase_admin import credentials, firestore

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SERVICE_ACCOUNT_PATH = os.path.join(BASE_DIR, "serviceAccountKey.json")

if not firebase_admin._apps:
    cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
    firebase_admin.initialize_app(cred)

db = firestore.client()


def list_cases():
    """
    List all cases stored in Firestore
    """
    docs = db.collection("cases").stream()

    rows = []

    for doc in docs:
        data = doc.to_dict()

        rows.append({
            "id": doc.id,
            "prediction": data.get("prediction", ""),
            "confidence": data.get("confidence", 0),
            "groundTruth": data.get("groundTruth", ""),
        })

    print("\nCases in Firestore:\n")

    for i, row in enumerate(rows, start=1):
        print(
            f"{i}. ID: {row['id']} | "
            f"Prediction: {row['prediction']} | "
            f"Confidence: {row['confidence']:.3f} | "
            f"GroundTruth: {row['groundTruth'] or 'EMPTY'}"
        )


def update_case_label(case_id: str, label: str):
    """
    Update groundTruth for a specific case
    """

    label = label.strip().capitalize()

    if label not in {"Fracture", "Normal"}:
        raise ValueError("Label must be 'Fracture' or 'Normal'")

    ref = db.collection("cases").document(case_id)
    snap = ref.get()

    if not snap.exists:
        print("Case not found")
        return

    ref.update({"groundTruth": label})

    print(f"Updated {case_id} -> groundTruth = {label}")


def fill_empty_labels(default_label="Normal"):
    """
    Fill all empty groundTruth labels
    """

    docs = db.collection("cases").stream()

    count = 0

    for doc in docs:
        data = doc.to_dict()

        if not data.get("groundTruth"):
            db.collection("cases").document(doc.id).update({
                "groundTruth": default_label
            })
            count += 1

    print(f"Updated {count} cases")


if __name__ == "__main__":

    print("\nMEDORA Label Tool\n")

    print("1 - Show all cases")
    print("2 - Update single case label")
    print("3 - Fill empty labels")

    choice = input("\nChoose option: ")

    if choice == "1":
        list_cases()

    elif choice == "2":

        case_id = input("Enter case ID: ")
        label = input("Enter label (Fracture / Normal): ")

        update_case_label(case_id, label)

    elif choice == "3":

        label = input("Default label (Fracture / Normal): ")
        fill_empty_labels(label)

    else:
        print("Invalid option")