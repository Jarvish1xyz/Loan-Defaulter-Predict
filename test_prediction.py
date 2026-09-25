import os
import json
import pickle as pkl
import pandas as pd
import numpy as np

def test_inference():
    print("Testing model artifacts...")

    required_models = [
        "logistic_regression_model.pkl",
        "decision_tree_model.pkl",
        "knn_model.pkl",
        "bagging_model.pkl",
        "boosting_model.pkl",
        "preprocessor.pkl",
    ]

    for model_name in required_models:
        file_path = os.path.join("models", model_name)
        assert os.path.exists(file_path), f"Missing required model file: {model_name}"
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"  [OK] Found {model_name} ({file_size_mb:.2f} MB)")
        # Ensure model is under 50MB (GitHub warning limit and safe for direct deployment)
        assert file_size_mb < 50.0, f"File {model_name} is too large ({file_size_mb:.2f} MB) for GitHub upload"

    def _load(name):
        with open(os.path.join("models", name), "rb") as f:
            return pkl.load(f)

    lr_model = _load("logistic_regression_model.pkl")
    dt_model = _load("decision_tree_model.pkl")
    knn_model = _load("knn_model.pkl")
    bagging_model = _load("bagging_model.pkl")
    boosting_model = _load("boosting_model.pkl")
    preprocessor = _load("preprocessor.pkl")

    with open("models/model_metadata.json") as f:
        meta = json.load(f)

    for key in ["logistic_regression", "decision_tree", "knn", "bagging", "boosting"]:
        assert key in meta["metrics"], f"Missing metric summary for {key}"
        m = meta["metrics"][key]
        for expected_metric in ["accuracy", "balanced_accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc", "log_loss", "confusion_matrix"]:
            assert expected_metric in m, f"Missing {expected_metric} in {key} metrics"

    thresholds = meta.get("thresholds", {}) or {}
    print("Metadata loaded successfully. Features count:", len(meta["all_feature_names"]))
    print("All Task 5 metrics (Accuracy, Balanced Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Log Loss, CM) verified.")

    # Test Case 1: Prime borrower
    prime_borrower = {
        "Age": 55, "Income": 140000, "LoanAmount": 30000, "CreditScore": 810,
        "MonthsEmployed": 100, "NumCreditLines": 1, "InterestRate": 4.5,
        "LoanTerm": 24, "DTIRatio": 0.15, "Education": "PhD",
        "EmploymentType": "Full-time", "MaritalStatus": "Married",
        "HasMortgage": "Yes", "HasDependents": "Yes", "LoanPurpose": "Home",
        "HasCoSigner": "Yes"
    }

    # Test Case 2: High risk applicant
    high_risk = {
        "Age": 21, "Income": 22000, "LoanAmount": 200000, "CreditScore": 420,
        "MonthsEmployed": 2, "NumCreditLines": 4, "InterestRate": 23.0,
        "LoanTerm": 60, "DTIRatio": 0.88, "Education": "High School",
        "EmploymentType": "Unemployed", "MaritalStatus": "Single",
        "HasMortgage": "No", "HasDependents": "Yes", "LoanPurpose": "Business",
        "HasCoSigner": "No"
    }

    df_test = pd.DataFrame([prime_borrower, high_risk])
    X_proc = preprocessor.transform(df_test)

    risk_models = {
        "LR": lr_model,
        "DT": dt_model,
        "KNN": knn_model,
        "BAGGING": bagging_model,
        "BOOSTING": boosting_model,
    }

    print("\n--- Test Results ---")
    for name, model in risk_models.items():
        probs = model.predict_proba(X_proc)[:, 1]
        threshold = thresholds.get(name.lower(), 0.50)
        preds = (probs >= threshold).astype(int)
        print(f"{name} prime risk: {probs[0]*100:.1f}% | high risk: {probs[1]*100:.1f}% | preds: {preds.tolist()}")
        assert probs[0] < probs[1], f"{name} risk should be lower for the prime borrower"
        assert preds[0] == 0 and preds[1] == 1, f"{name} verdicts should reject the high-risk applicant"

    print("\n[SUCCESS] Model predictions verified and consistent!")

if __name__ == "__main__":
    test_inference()
