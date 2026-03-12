from typing import Any, List, Dict


def normalize_binary_label(value: Any):
    """
    Convert labels into binary:
    Fracture / Positive / 1 / True -> 1
    Normal / Negative / 0 / False -> 0
    """
    if value is None:
        return None

    v = str(value).strip().lower()

    if v in {"fracture", "positive", "1", "true"}:
        return 1
    if v in {"normal", "negative", "0", "false"}:
        return 0

    return None


def compute_confusion_counts(rows: List[Dict]):
    tp = tn = fp = fn = 0

    for row in rows:
        gt = row["gt"]
        pred = row["pred"]

        if gt == 1 and pred == 1:
            tp += 1
        elif gt == 0 and pred == 0:
            tn += 1
        elif gt == 0 and pred == 1:
            fp += 1
        elif gt == 1 and pred == 0:
            fn += 1

    return tp, tn, fp, fn


def compute_roc_curve(rows: List[Dict]):
    thresholds = [i / 100 for i in range(101)]
    points = []

    for threshold in thresholds:
        temp = []

        for row in rows:
            pred = 1 if row["score"] >= threshold else 0
            temp.append({
                "gt": row["gt"],
                "pred": pred,
                "score": row["score"]
            })

        tp, tn, fp, fn = compute_confusion_counts(temp)

        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0

        points.append({
            "threshold": round(threshold, 2),
            "tpr": tpr,
            "fpr": fpr
        })

    return points


def compute_pr_curve(rows: List[Dict]):
    thresholds = [i / 100 for i in range(101)]
    points = []

    for threshold in thresholds:
        tp = fp = fn = 0

        for row in rows:
            pred = 1 if row["score"] >= threshold else 0
            gt = row["gt"]

            if gt == 1 and pred == 1:
                tp += 1
            elif gt == 0 and pred == 1:
                fp += 1
            elif gt == 1 and pred == 0:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0

        points.append({
            "threshold": round(threshold, 2),
            "precision": precision,
            "recall": recall
        })

    return points


def compute_confidence_histogram(rows: List[Dict]):
    bins = [
        {"range": "0.0-0.2", "count": 0},
        {"range": "0.2-0.4", "count": 0},
        {"range": "0.4-0.6", "count": 0},
        {"range": "0.6-0.8", "count": 0},
        {"range": "0.8-1.0", "count": 0},
    ]

    for row in rows:
        score = float(row.get("score", 0.0))

        if score < 0.2:
            bins[0]["count"] += 1
        elif score < 0.4:
            bins[1]["count"] += 1
        elif score < 0.6:
            bins[2]["count"] += 1
        elif score < 0.8:
            bins[3]["count"] += 1
        else:
            bins[4]["count"] += 1

    return bins


def compute_binary_metrics(rows: List[Dict]):
    """
    rows input example:
    [
        {
            "groundTruth": "Fracture",
            "prediction": "Fracture",
            "score": 0.88
        }
    ]
    """
    evaluated = []

    for row in rows:
        gt = normalize_binary_label(row.get("groundTruth"))
        pred = normalize_binary_label(row.get("prediction"))
        score = float(row.get("score", 0.0) or 0.0)

        if gt is None:
            continue

        if pred is None:
            pred = 1 if score >= 0.5 else 0

        evaluated.append({
            "gt": gt,
            "pred": pred,
            "score": score
        })

    tp, tn, fp, fn = compute_confusion_counts(evaluated)

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "evaluatedCount": total,
        "confusionMatrix": {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn
        },
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1
        },
        "rocCurve": compute_roc_curve(evaluated),
        "prCurve": compute_pr_curve(evaluated),
        "confidenceHistogram": compute_confidence_histogram(evaluated)
    }