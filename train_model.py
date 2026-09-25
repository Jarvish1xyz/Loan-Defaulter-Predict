"""
Comprehensive Model Training, Evaluation, 5-Fold Cross-Validation, Overfitting Diagnostics,
Hyperparameter Tuning, and ROC/Log-Loss Analysis Pipeline for Loan Default Detection (~255K rows).

Implements all checklist tasks from Task 5 document:
  1. Full classification evaluation: Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC, Log Loss, Confusion Matrix.
  2. Overfitting / Underfitting diagnostics: Train Score vs. Test Score comparison.
  3. 5-Fold Stratified Cross-Validation on training data with Mean & Std Spread.
  4. Model comparison and ranking table.
  5. Hyperparameter tuning via RandomizedSearchCV with Stratified CV.
  6. Advanced model architectures: Logistic Regression, Decision Tree, Bagging (Random Forest), Boosting, and KNN.
  7. Extraction of ROC curves and PR curves for rich interactive dashboard visualization.
"""

import os
import json
import time
import pickle as pkl
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from imblearn.over_sampling import SMOTE
from preprocess import (
    NUMERICAL_COLS,
    CATEGORICAL_COLS,
    ENGINEERED_COLS,
    build_preprocessor,
)


def tune_threshold(prob_pos, y_true, grid_step=0.01):
    """Find the probability threshold maximising balanced accuracy on validation data."""
    grid = np.arange(0.05, 0.96, grid_step)
    best_threshold = 0.50
    best_score = -1.0
    for threshold in grid:
        preds = (prob_pos >= threshold).astype(int)
        score = balanced_accuracy_score(y_true, preds)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def evaluate_model_full(model, threshold, X_eval, y_eval, X_train=None, y_train=None):
    """Evaluate model with full Task 5 suite: Acc, Prec, Rec, F1, ROC-AUC, PR-AUC, Log-Loss, Overfitting."""
    prob_pos = model.predict_proba(X_eval)[:, 1]
    prob_pos_clipped = np.clip(prob_pos, 1e-15, 1 - 1e-15)
    preds = (prob_pos >= threshold).astype(int)

    acc = accuracy_score(y_eval, preds)
    bal_acc = balanced_accuracy_score(y_eval, preds)
    prec = precision_score(y_eval, preds, zero_division=0)
    rec = recall_score(y_eval, preds, zero_division=0)
    f1 = f1_score(y_eval, preds, zero_division=0)
    roc_auc = roc_auc_score(y_eval, prob_pos)
    pr_auc = average_precision_score(y_eval, prob_pos)
    loss = log_loss(y_eval, prob_pos_clipped)
    cm = confusion_matrix(y_eval, preds).tolist()

    # Calculate ROC Curve & PR Curve points (downsampled to 100 points for dashboard JSON compactness)
    fpr, tpr, _ = roc_curve(y_eval, prob_pos)
    pr_prec, pr_rec, _ = precision_recall_curve(y_eval, prob_pos)

    step_roc = max(1, len(fpr) // 100)
    step_pr = max(1, len(pr_prec) // 100)

    roc_curve_data = {
        "fpr": [round(float(x), 4) for x in fpr[::step_roc]] + [1.0],
        "tpr": [round(float(x), 4) for x in tpr[::step_roc]] + [1.0],
    }
    pr_curve_data = {
        "recall": [round(float(x), 4) for x in pr_rec[::step_pr]],
        "precision": [round(float(x), 4) for x in pr_prec[::step_pr]],
    }

    # Overfitting / Underfitting check (Train vs Test score)
    train_metrics = {}
    fit_status = "Good Fit"
    if X_train is not None and y_train is not None:
        train_prob = model.predict_proba(X_train)[:, 1]
        train_preds = (train_prob >= threshold).astype(int)
        train_acc = accuracy_score(y_train, train_preds)
        train_f1 = f1_score(y_train, train_preds, zero_division=0)
        train_auc = roc_auc_score(y_train, train_prob)
        train_metrics = {
            "train_accuracy": round(float(train_acc), 4),
            "train_f1": round(float(train_f1), 4),
            "train_roc_auc": round(float(train_auc), 4),
        }
        gap = train_acc - acc
        if gap > 0.12:
            fit_status = "Overfitting (Train >> Test)"
        elif gap > 0.05:
            fit_status = "Slight Overfit"
        elif acc < 0.60 and train_acc < 0.60:
            fit_status = "Underfitting (Both Low)"
        else:
            fit_status = "Good Fit"

    return {
        "accuracy": round(float(acc), 4),
        "balanced_accuracy": round(float(bal_acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "log_loss": round(float(loss), 4),
        "confusion_matrix": cm,
        "roc_curve": roc_curve_data,
        "pr_curve": pr_curve_data,
        "train_metrics": train_metrics,
        "fit_status": fit_status,
    }


def perform_5fold_cross_validation(models_unfitted, X_train_proc, y_train):
    """Run 5-Fold Stratified Cross-Validation on training data and return Mean & Spread in a single pass."""
    print("\n--- Step 3: Running 5-Fold Stratified Cross-Validation ---")
    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}

    # Fast stratified sample for efficient, stable CV across 255k dataset
    sub_idx, _ = train_test_split(
        np.arange(len(y_train)),
        train_size=min(30000, len(y_train)),
        stratify=y_train,
        random_state=42,
    )
    X_cv = X_train_proc[sub_idx]
    y_cv = y_train.iloc[sub_idx] if hasattr(y_train, 'iloc') else y_train[sub_idx]

    scoring = {"roc_auc": "roc_auc", "accuracy": "accuracy", "f1": "f1"}

    for name, model in models_unfitted.items():
        print(f"   Evaluating 5-Fold CV for {name.upper()}...")
        res = cross_validate(model, X_cv, y_cv, cv=cv5, scoring=scoring, n_jobs=-1)

        scores_roc = res["test_roc_auc"]
        scores_acc = res["test_accuracy"]
        scores_f1 = res["test_f1"]

        cv_results[name] = {
            "cv_roc_auc_mean": round(float(scores_roc.mean()), 4),
            "cv_roc_auc_std": round(float(scores_roc.std()), 4),
            "cv_accuracy_mean": round(float(scores_acc.mean()), 4),
            "cv_accuracy_std": round(float(scores_acc.std()), 4),
            "cv_f1_mean": round(float(scores_f1.mean()), 4),
            "cv_f1_std": round(float(scores_f1.std()), 4),
            "cv_folds_roc_auc": [round(float(s), 4) for s in scores_roc],
        }
        print(f"      -> ROC-AUC: {scores_roc.mean():.4f} +/- {scores_roc.std():.4f} | Acc: {scores_acc.mean():.4f} +/- {scores_acc.std():.4f}")

    return cv_results


def tune_and_train_all_models(X_train_proc, y_train, X_train_smote, y_train_smote):
    """Hyperparameter Tuning with StratifiedKFold CV and fitting tuned models."""
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    models = {}
    best_params = {}

    # Representative stratified sample for fast hyperparameter search
    tune_idx, _ = train_test_split(
        np.arange(len(y_train)),
        train_size=min(35000, len(y_train)),
        stratify=y_train,
        random_state=42,
    )
    X_tune = X_train_proc[tune_idx]
    y_tune = y_train.iloc[tune_idx] if hasattr(y_train, 'iloc') else y_train[tune_idx]

    # 1. LOGISTIC REGRESSION
    print("\n[1/5] Hyperparameter Tuning: LOGISTIC REGRESSION...")
    lr_search = RandomizedSearchCV(
        LogisticRegression(max_iter=1000, random_state=42),
        {
            "C": [0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
            "solver": ["lbfgs", "saga"],
            "class_weight": ["balanced", None],
        },
        n_iter=5,
        scoring="roc_auc",
        cv=cv,
        random_state=42,
        n_jobs=-1,
    )
    lr_search.fit(X_tune, y_tune)
    best_lr = lr_search.best_params_
    print(f"   [BEST] LR Params: {best_lr} (CV ROC-AUC: {lr_search.best_score_:.4f})")
    
    final_lr = LogisticRegression(
        C=best_lr.get("C", 0.5),
        solver=best_lr.get("solver", "lbfgs"),
        class_weight=best_lr.get("class_weight", "balanced"),
        max_iter=2000,
        random_state=42,
    )
    final_lr.fit(X_train_smote, y_train_smote)
    models["logistic_regression"] = final_lr
    best_params["logistic_regression"] = best_lr

    # 2. DECISION TREE
    print("\n[2/5] Hyperparameter Tuning: DECISION TREE...")
    dt_search = RandomizedSearchCV(
        DecisionTreeClassifier(random_state=42),
        {
            "max_depth": [6, 8, 10, 12],
            "min_samples_split": [50, 100, 200],
            "min_samples_leaf": [20, 40, 80],
            "criterion": ["gini", "entropy"],
            "class_weight": ["balanced", None],
        },
        n_iter=6,
        scoring="roc_auc",
        cv=cv,
        random_state=42,
        n_jobs=-1,
    )
    dt_search.fit(X_tune, y_tune)
    best_dt = dt_search.best_params_
    print(f"   [BEST] DT Params: {best_dt} (CV ROC-AUC: {dt_search.best_score_:.4f})")
    
    final_dt = DecisionTreeClassifier(**best_dt, random_state=42)
    final_dt.fit(X_train_proc, y_train)
    models["decision_tree"] = final_dt
    best_params["decision_tree"] = best_dt

    # 3. BAGGING / RANDOM FOREST
    print("\n[3/5] Hyperparameter Tuning: BAGGING (RANDOM FOREST)...")
    rf_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        {
            "n_estimators": [60, 80, 100],
            "max_depth": [10, 12, 14],
            "min_samples_leaf": [15, 25, 40],
            "max_features": ["sqrt", 0.6],
            "class_weight": ["balanced", None],
        },
        n_iter=5,
        scoring="roc_auc",
        cv=cv,
        random_state=42,
        n_jobs=-1,
    )
    rf_search.fit(X_tune, y_tune)
    best_rf = rf_search.best_params_
    print(f"   [BEST] Bagging/RF Params: {best_rf} (CV ROC-AUC: {rf_search.best_score_:.4f})")
    
    final_rf = RandomForestClassifier(**best_rf, random_state=42, n_jobs=-1)
    final_rf.fit(X_train_proc, y_train)
    models["bagging"] = final_rf
    best_params["bagging"] = best_rf

    # 4. BOOSTING (HIST GRADIENT BOOSTING)
    print("\n[4/5] Hyperparameter Tuning: BOOSTING (HIST GRADIENT BOOSTING)...")
    boost_search = RandomizedSearchCV(
        HistGradientBoostingClassifier(random_state=42),
        {
            "learning_rate": [0.05, 0.08, 0.12],
            "max_iter": [120, 180, 240],
            "max_depth": [5, 6, 8],
            "max_leaf_nodes": [20, 31],
            "min_samples_leaf": [30, 50, 100],
            "l2_regularization": [0.0, 0.5, 1.5],
            "class_weight": ["balanced", None],
        },
        n_iter=5,
        scoring="roc_auc",
        cv=cv,
        random_state=42,
        n_jobs=-1,
    )
    boost_search.fit(X_tune, y_tune)
    best_boost = boost_search.best_params_
    print(f"   [BEST] Boosting Params: {best_boost} (CV ROC-AUC: {boost_search.best_score_:.4f})")
    
    final_boost = HistGradientBoostingClassifier(
        **best_boost,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
    )
    final_boost.fit(X_train_proc, y_train)
    models["boosting"] = final_boost
    best_params["boosting"] = best_boost

    # 5. KNN (Root N Heuristic for n_neighbors + GitHub deployable compact sizing)
    print("\n[5/5] Hyperparameter Tuning: KNN (Root N heuristic & GitHub deployment sizing)...")
    knn_sample_size = 25000
    sub_idx, _ = train_test_split(
        np.arange(len(y_train)),
        train_size=min(knn_sample_size, len(y_train)),
        stratify=y_train,
        random_state=42,
    )
    X_train_knn = np.ascontiguousarray(X_train_proc[sub_idx], dtype=np.float32)
    y_train_knn = y_train.iloc[sub_idx] if hasattr(y_train, 'iloc') else y_train[sub_idx]

    # Calculate root N for n_neighbors (odd integer to avoid tied voting)
    n_knn_samples = len(X_train_knn)
    k_root_n = int(np.round(np.sqrt(n_knn_samples)))
    if k_root_n % 2 == 0:
        k_root_n += 1
    print(f"   [INFO] KNN training sample: N={n_knn_samples:,} | Root N = sqrt({n_knn_samples}) -> k_root_n = {k_root_n}")

    candidate_k = sorted(list(set([
        max(3, (k_root_n - 40) | 1),
        max(3, (k_root_n - 20) | 1),
        k_root_n,
        (k_root_n + 20) | 1,
        (k_root_n + 40) | 1
    ])))

    knn_search = RandomizedSearchCV(
        KNeighborsClassifier(n_jobs=-1),
        {
            "n_neighbors": candidate_k,
            "weights": ["uniform", "distance"],
            "metric": ["minkowski", "manhattan"],
        },
        n_iter=5,
        scoring="roc_auc",
        cv=cv,
        random_state=42,
        n_jobs=-1,
    )
    knn_search.fit(X_train_knn, y_train_knn)
    best_knn = knn_search.best_params_
    best_knn["root_n_base"] = k_root_n
    print(f"   [BEST] KNN Params: {best_knn} (CV ROC-AUC: {knn_search.best_score_:.4f})")
    
    final_knn = KNeighborsClassifier(
        n_neighbors=best_knn.get("n_neighbors", k_root_n),
        weights=best_knn.get("weights", "distance"),
        metric=best_knn.get("metric", "minkowski"),
        n_jobs=-1
    )
    final_knn.fit(X_train_knn, y_train_knn)
    models["knn"] = final_knn
    best_params["knn"] = best_knn

    return models, best_params


def calculate_composite_score(metrics_dict):
    """Compute fair composite ranking score including PR-AUC."""
    return round(
        float(
            metrics_dict["balanced_accuracy"] * 0.20
            + metrics_dict["recall"] * 0.20
            + metrics_dict["pr_auc"] * 0.20
            + metrics_dict["roc_auc"] * 0.15
            + metrics_dict["f1_score"] * 0.15
            + metrics_dict["precision"] * 0.05
            + metrics_dict["accuracy"] * 0.05
        ),
        4
    )


def main():
    start_time = time.time()
    print("=" * 75)
    print("FINSECURE AI - ADVANCED LOAN DEFAULT PIPELINE (TASK 5 FULL SPECIFICATION)")
    print("=" * 75)

    os.makedirs("models", exist_ok=True)

    data_path = "Loan_default.csv"
    print(f"Loading dataset from '{data_path}'...")
    df = pd.read_csv(data_path)
    total_rows, total_cols = df.shape
    print(f"[OK] Dataset loaded: {total_rows:,} rows, {total_cols} columns.")

    target_col = "Default"
    id_col = "LoanID"

    if id_col in df.columns:
        df = df.drop(columns=[id_col])
        print(f"[OK] Dropped identifier column '{id_col}'.")

    X = df.drop(columns=[target_col])
    y = df[target_col]
    default_rate = float(y.mean())
    print(f"[OK] Target distribution: Non-Default={(y==0).sum():,} | Default={(y==1).sum():,} ({default_rate:.2%})")

    # Step 1: Stratified Train / Val / Test Split (64% / 16% / 20%)
    print("\n--- Step 1: Stratified Partitioning (64% Train, 16% Val, 20% Test) ---")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.36, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(20.0 / 36.0), random_state=42, stratify=y_temp
    )

    print(f"[OK] Training set:     {X_train.shape[0]:,} samples ({X_train.shape[0]/total_rows:.1%})")
    print(f"[OK] Validation set:   {X_val.shape[0]:,} samples ({X_val.shape[0]/total_rows:.1%})")
    print(f"[OK] Testing set:      {X_test.shape[0]:,} samples ({X_test.shape[0]/total_rows:.1%}) [Untouched]")

    # Step 2: Build Preprocessor & Transform (fitted ONLY on X_train)
    print("\n--- Step 2: Preprocessing Pipeline (Fitted strictly on Training Set) ---")
    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    cat_encoder = preprocessor.named_transformers_["cat"]
    encoded_cat_cols = list(cat_encoder.get_feature_names_out(CATEGORICAL_COLS))
    all_feature_names = NUMERICAL_COLS + ENGINEERED_COLS + encoded_cat_cols
    print(f"[OK] Preprocessor fitted. Total feature dimensions: {len(all_feature_names)}")

    # Step 3: Class Imbalance Resampling
    print("\n--- Step 3: Imbalance Handling (SMOTE on Training data only) ---")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_proc, y_train)
    print(f"[OK] SMOTE resampled training set: {X_train_smote.shape[0]:,} samples (50/50 balance)")

    # Step 4: Hyperparameter Tuning & Model Fitting
    print("\n--- Step 4: Hyperparameter Tuning with StratifiedKFold CV ---")
    models, best_hyperparams = tune_and_train_all_models(
        X_train_proc, y_train, X_train_smote, y_train_smote
    )

    # Step 5: 5-Fold Cross-Validation on Training Set
    knn_cv_params = {k: v for k, v in best_hyperparams["knn"].items() if k != "root_n_base"}
    unfitted_prototypes = {
        "logistic_regression": LogisticRegression(**best_hyperparams["logistic_regression"], max_iter=1000, random_state=42),
        "decision_tree": DecisionTreeClassifier(**best_hyperparams["decision_tree"], random_state=42),
        "bagging": RandomForestClassifier(**best_hyperparams["bagging"], random_state=42, n_jobs=-1),
        "boosting": HistGradientBoostingClassifier(**best_hyperparams["boosting"], random_state=42),
        "knn": KNeighborsClassifier(**knn_cv_params, n_jobs=-1),
    }
    cv_5fold_results = perform_5fold_cross_validation(unfitted_prototypes, X_train_proc, y_train)

    # Step 6: Threshold Optimization on Validation Split
    print("\n--- Step 6: Decision Threshold Optimization on Validation Split ---")
    thresholds = {}
    val_metrics = {}
    for name, model in models.items():
        val_probs = model.predict_proba(X_val_proc)[:, 1]
        best_th = tune_threshold(val_probs, y_val)
        thresholds[name] = round(best_th, 4)
        val_metrics[name] = evaluate_model_full(model, best_th, X_val_proc, y_val)
        print(f"   - {name.upper():20s}: Optimal Threshold = {thresholds[name]:.2f} | Val BalAcc = {val_metrics[name]['balanced_accuracy']:.4f} | Val PR-AUC = {val_metrics[name]['pr_auc']:.4f} | Log Loss = {val_metrics[name]['log_loss']:.4f}")

    # Step 7: Final One-Time Evaluation on Untouched Test Set
    print("\n--- Step 7: Final Evaluation on Untouched Test Set (51,070 records) ---")
    test_metrics = {}
    ranking_summary = []

    # Small sample of X_train for fast overfitting metric computation
    train_eval_idx, _ = train_test_split(np.arange(len(y_train)), train_size=25000, stratify=y_train, random_state=42)
    X_tr_eval = X_train_proc[train_eval_idx]
    y_tr_eval = y_train.iloc[train_eval_idx] if hasattr(y_train, 'iloc') else y_train[train_eval_idx]

    for name, model in models.items():
        th = thresholds[name]
        m = evaluate_model_full(model, th, X_test_proc, y_test, X_tr_eval, y_tr_eval)
        m["test_samples"] = int(len(y_test))
        m["decision_threshold"] = th
        m["composite_score"] = calculate_composite_score(m)
        m["cv_5fold"] = cv_5fold_results.get(name, {})
        test_metrics[name] = m

        ranking_summary.append({
            "model_key": name,
            "model_name": name.replace("_", " ").title(),
            "composite_score": m["composite_score"],
            "accuracy": m["accuracy"],
            "train_accuracy": m["train_metrics"].get("train_accuracy", m["accuracy"]),
            "balanced_accuracy": m["balanced_accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1_score": m["f1_score"],
            "roc_auc": m["roc_auc"],
            "pr_auc": m["pr_auc"],
            "log_loss": m["log_loss"],
            "fit_status": m["fit_status"],
            "cv_roc_mean": m["cv_5fold"].get("cv_roc_auc_mean", 0.0),
            "cv_roc_std": m["cv_5fold"].get("cv_roc_auc_std", 0.0),
        })

        print(f"\n[{name.upper()}] (Threshold = {th:.2f})")
        print(f"   Accuracy:          {m['accuracy']:.4f} (Train: {m['train_metrics'].get('train_accuracy', 0):.4f} -> {m['fit_status']})")
        print(f"   Balanced Accuracy: {m['balanced_accuracy']:.4f}")
        print(f"   Precision:         {m['precision']:.4f}")
        print(f"   Recall:            {m['recall']:.4f}")
        print(f"   F1-Score:          {m['f1_score']:.4f}")
        print(f"   ROC-AUC:           {m['roc_auc']:.4f}")
        print(f"   PR-AUC:            {m['pr_auc']:.4f}")
        print(f"   Log Loss:          {m['log_loss']:.4f}")
        print(f"   Composite Score:   {m['composite_score']:.4f}")

    # Rank all models
    ranking_summary = sorted(ranking_summary, key=lambda x: x["composite_score"], reverse=True)
    best_model_key = ranking_summary[0]["model_key"]
    best_model_name = ranking_summary[0]["model_name"]
    print("\n" + "=" * 75)
    print(f"[BEST] MODEL RANKING SUMMARY (Best Model: {best_model_name.upper()})")
    print("=" * 75)
    for rank, item in enumerate(ranking_summary, start=1):
        print(f"#{rank} {item['model_name']:20s} | Score: {item['composite_score']:.4f} | Acc: {item['accuracy']:.4f} | PR-AUC: {item['pr_auc']:.4f} | ROC-AUC: {item['roc_auc']:.4f} | LogLoss: {item['log_loss']:.4f} | Fit: {item['fit_status']}")

    # Step 8: Feature Importance / Coefficients Analysis
    print("\n--- Step 8: Feature Importance Extraction ---")
    lr_model = models["logistic_regression"]
    dt_model = models["decision_tree"]
    bagging_model = models["bagging"]
    boosting_model = models["boosting"]

    lr_coefs = lr_model.coef_[0].tolist()
    dt_importances = dt_model.feature_importances_.tolist()
    rf_importances = bagging_model.feature_importances_.tolist()
    
    feature_analysis = {
        "feature_names": all_feature_names,
        "logistic_regression_coefficients": dict(
            zip(all_feature_names, [round(float(c), 4) for c in lr_coefs])
        ),
        "decision_tree_importances": dict(
            zip(all_feature_names, [round(float(i), 4) for i in dt_importances])
        ),
        "bagging_importances": dict(
            zip(all_feature_names, [round(float(i), 4) for i in rf_importances])
        ),
    }

    try:
        boosting_imp = boosting_model.feature_importances_.tolist()
        feature_analysis["boosting_importances"] = dict(
            zip(all_feature_names, [round(float(i), 4) for i in boosting_imp])
        )
    except Exception:
        feature_analysis["boosting_importances"] = {}

    # Numerical statistics for UI
    stats = {}
    for col in NUMERICAL_COLS:
        stats[col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": float(df[col].mean()),
            "median": float(df[col].median()),
            "std": float(df[col].std()),
        }

    categorical_options = {}
    for col in CATEGORICAL_COLS:
        categorical_options[col] = sorted(df[col].unique().tolist())

    metadata = {
        "dataset_name": "Loan Default Dataset",
        "total_rows": len(df),
        "numerical_columns": NUMERICAL_COLS,
        "categorical_columns": CATEGORICAL_COLS,
        "engineered_columns": ENGINEERED_COLS,
        "all_feature_names": all_feature_names,
        "metrics": test_metrics,
        "validation_metrics": val_metrics,
        "cv_5fold_results": cv_5fold_results,
        "thresholds": thresholds,
        "best_hyperparameters": best_hyperparams,
        "ranking_summary": ranking_summary,
        "best_model": {
            "key": best_model_key,
            "name": best_model_name,
            "composite_score": ranking_summary[0]["composite_score"],
        },
        "feature_analysis": feature_analysis,
        "numerical_stats": stats,
        "categorical_options": categorical_options,
        "target_distribution": {
            "non_default_0": int((df[target_col] == 0).sum()),
            "default_1": int((df[target_col] == 1).sum()),
            "default_rate": round(float(df[target_col].mean()), 4),
        },
        "partition_sizes": {
            "train": len(y_train),
            "val": len(y_val),
            "test": len(y_test),
        }
    }

    # Step 9: Save All Artifacts
    print("\n--- Step 9: Saving Serialized Model Artifacts (.pkl format for GitHub deployment) ---")

    def save_pkl_artifact(obj, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pkl.dump(obj, f, protocol=pkl.HIGHEST_PROTOCOL)
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"[OK] Saved '{filepath}' ({size_mb:.2f} MB) - GitHub upload ready!")

    save_pkl_artifact(lr_model, "models/logistic_regression_model.pkl")
    save_pkl_artifact(dt_model, "models/decision_tree_model.pkl")
    save_pkl_artifact(models["knn"], "models/knn_model.pkl")
    save_pkl_artifact(bagging_model, "models/bagging_model.pkl")
    save_pkl_artifact(boosting_model, "models/boosting_model.pkl")
    save_pkl_artifact(preprocessor, "models/preprocessor.pkl")

    with open("models/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    print("[OK] Saved 'models/model_metadata.json'")

    sample_test_df = X_test.copy()
    sample_test_df["Actual_Default"] = y_test.values
    sample_test_df.head(200).to_csv("models/sample_test_data.csv", index=False)
    print("[OK] Saved 'models/sample_test_data.csv' (200 test samples)")

    elapsed = time.time() - start_time
    print("\n" + "=" * 75)
    print(f"[SUCCESS] Complete Task 5 Pipeline Finished in {elapsed:.1f}s!")
    print("=" * 75)


if __name__ == "__main__":
    main()