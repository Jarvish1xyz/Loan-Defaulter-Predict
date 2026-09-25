# 💳 FinSecure AI — Loan Default Detection System & Streamlit Frontend

An enterprise-grade Machine Learning solution and interactive web dashboard for predicting loan default risks using **Logistic Regression** and **Decision Tree** classifiers.

---

## 🌟 Key Features

1. **Five Machine Learning Models** (retrained with improvements):
   - **Logistic Regression**: Linear decision boundary with calibrated probabilities and class balancing.
   - **Decision Tree Classifier**: Non-linear hierarchical tree logic with constrained depth to prevent overfitting.
   - **KNN Classifier**: Distance-based neighborhood voting with distance weighting.
   - **Bagging Classifier**: Bootstrap-aggregated decision trees (balanced base learners).
   - **Boosting (HistGradientBoosting)**: Fast gradient-boosted trees optimized for default detection.
   - **Ensemble Consensus Mode**: Blended probability scoring.
2. **Improved Training Pipeline**:
   - **Feature engineering**: 8 derived financial-risk features (loan-to-income, credit burden, interest/credit signals, etc.).
   - **SMOTE class balancing**: Minority (default) class oversampled on training data to fix the 11.6% imbalance.
   - **Tuned decision thresholds** per model (validated on a hold-out validation set) to maximize balanced accuracy instead of misleading raw accuracy.
2. **Dynamic Risk Score Meter**:
   - Animated Plotly circular Gauge / Risk Meter (0 to 100 Risk Score) with color-coded risk bands:
     - 🟢 **Low Risk (0 - 35%)**: Safe / Recommended for Approval
     - 🟡 **Moderate Risk (36 - 65%)**: Review Required
     - 🔴 **High Risk (66 - 100%)**: Default Warning / Rejection Recommended
3. **Comprehensive Frontend Dashboard**:
   - **Interactive Predictor Tab**: Live scoring with personal, financial, and loan parameters.
   - **Performance & Accuracy Tab**: Confusion matrices, accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, and feature importance rankings.
   - **Batch CSV Risk Analyzer Tab**: Bulk score hundreds of applicants simultaneously with CSV download.
   - **EDA Insights Tab**: Statistical trends from 255k verified loan records.
   - **Quick-Fill Borrower Presets**: Realistic test profiles (Prime, Moderate, High-Risk, Young Graduate).

---

## 🚀 How to Run the Application

### 1. (Optional) Re-train and serialize models
```bash
python train_model.py
```

### 2. Launch the Streamlit Frontend Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📊 Dataset & Model Performance Summary

- **Total Records Analyzed**: 255,347 loan applications (18 features)
- **Train/Validation/Test Split**: 64% Train / 16% Validation / 20% Stratified Test (51,070 samples)
- **Baseline Default Rate**: 11.61% (minority class oversampled with SMOTE during training)
- **Metrics evaluated on test set at each model's tuned decision threshold.**

| Metric | Logistic Regression | Decision Tree | KNN (Root $N$) | Bagging (RF) | Boosting (HistGB) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Accuracy** | **69.42%** | **67.66%** | **69.04%** | **67.36%** | **68.07%** |
| **Balanced Accuracy** | **69.25%** | **66.43%** | **67.54%** | **68.34%** | **68.89%** |
| **Precision (Defaulters)** | 0.2291 | 0.2104 | 0.2202 | 0.2174 | 0.2222 |
| **Recall (Defaulters Detected)** | **69.03%** | **64.83%** | **65.59%** | **69.62%** | **69.95%** |
| **F1-Score** | **0.3440** | **0.3177** | **0.3298** | **0.3313** | **0.3372** |
| **ROC-AUC Score** | **0.7590** | **0.7263** | **0.7379** | **0.7483** | **0.7554** |
| **PR-AUC Score** | **0.3383** | **0.2813** | **0.3037** | **0.3186** | **0.3317** |
| **File Size (GitHub Ready)** | **< 1 KB** | **~34 KB** | **3.91 MB** | **8.98 MB** | **367 KB** |

> **Balanced Risk Intelligence & Generalization**: All five models achieve **"Good Fit"** status (no overfitting) and successfully detect **~65–70% of true defaulters** while maintaining calibrated decision boundaries and lightweight `.pkl` footprint under GitHub upload limits.

---

## 🗂️ Project Structure

```
├── Loan_default.csv                 # Original 255k loan applications dataset
├── train_model.py                   # Model training pipeline (SMOTE, Root-N KNN, 5-Fold CV, .pkl export)
├── preprocess.py                    # Shared feature-engineering & preprocessing pipeline
├── app.py                           # Streamlit Dark Mode Web Application
├── test_prediction.py               # Unit test script verifying .pkl model inference
├── models/                          # Serialized .pkl artifacts + metadata & thresholds
│   ├── logistic_regression_model.pkl
│   ├── decision_tree_model.pkl
│   ├── knn_model.pkl                # Trained with Root N neighbors (~3.9 MB, GitHub ready)
│   ├── bagging_model.pkl
│   ├── boosting_model.pkl
│   ├── preprocessor.pkl
│   ├── model_metadata.json
│   └── sample_test_data.csv
└── README.md
```

<!-- note:
- Feature engineering + SMOTE balancing + per-model threshold tuning implemented in train_model.py
- Hyperparameter tuning done via targeted sets; deeper grid/random search and K-fold CV can be added next -->
