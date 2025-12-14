# Capstone-Project - Credit Card Fraud Detection using Machine Learning
Capstone Project - Credit Card Fraud Detection

## 📌 Project Overview
This project focuses on detecting fraudulent credit card transactions using machine learning techniques on a highly imbalanced dataset. The goal is to build an accurate, explainable, and production-ready fraud detection system.

---

## 📂 Dataset
- Source: Public credit card transaction dataset
- Features: 28 anonymized PCA-transformed features (`V1`–`V28`) + `Amount`
- Target:
  - `0` → Legitimate transaction
  - `1` → Fraudulent transaction
- Class imbalance: Fraud transactions constitute <0.2% of total data

---

## 🛠️ Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- SHAP (Explainable AI)
- Jupyter Notebook

---

## ⚙️ Machine Learning Pipeline
1. Data loading and exploration
2. Handling class imbalance
3. Feature scaling
4. Model training using Random Forest
5. Hyperparameter tuning with GridSearchCV
6. Model evaluation using:
   - Confusion Matrix
   - Precision, Recall, F1-score
   - ROC-AUC
7. Model explainability using SHAP

---

## 1. Project Objective

The objective of this project was to build a robust and explainable **Credit Card Fraud Detection system** capable of handling **highly imbalanced transaction data**, while maximizing the ability to detect fraudulent transactions (**Recall**) without causing excessive false alarms (**Precision**).

Multiple machine learning models were evaluated:
- Logistic Regression
- Random Forest
- XGBoost  

Each model was tested under:
- **Imbalanced data**
- **Balanced data using SMOTE**
- **Hyperparameter tuning**
- **Threshold optimization**
- **Precision-Recall based evaluation**

---

## 2. Key Challenges Addressed

- Severe **class imbalance** (fraud cases < 0.2%)
- Accuracy being a misleading metric
- Trade-off between **fraud detection (Recall)** and **customer inconvenience (Precision)**
- Computational constraints (CPU-only environment)
- Need for **explainability** in financial models

---

## 3. Model-wise Performance Summary

### 🔹 Logistic Regression
- **Imbalanced Data**:  
  - High accuracy but extremely poor recall
  - Failed to detect most fraud cases

- **SMOTE + Hyperparameter Tuning**:
  - Recall improved significantly
  - Precision dropped sharply due to many false positives
  - Suitable as a **baseline model**, but not ideal for production fraud systems

**Conclusion**:  
Logistic Regression benefits from SMOTE but struggles to balance precision and recall effectively.

---

### 🔹 Random Forest
- **Imbalanced Data (Class Weights)**:
  - Strong improvement in fraud detection
  - Better precision than Logistic Regression
  - Robust to imbalance without oversampling

- **SMOTE + Tuning**:
  - Recall improved, but precision degraded
  - SMOTE introduced noise for tree-based learning

**Conclusion**:  
Random Forest performs well with **cost-sensitive learning**, but SMOTE does not consistently improve results.

---

### 🔹 XGBoost (Best Performing Model)
- **Imbalanced Data with `scale_pos_weight`**:
  - Excellent Precision-Recall balance
  - Strong ranking ability (high PR-AUC)

- **SMOTE**:
  - Performance degraded due to synthetic noise
  - Tree-based boosting learned artificial patterns

- **Final Tuned Model (No SMOTE + Threshold Optimization)**:
  - **Precision: ~94%**
  - **Recall: ~81%**
  - **F1-score: ~0.87**
  - **PR-AUC: ~0.87**

**Conclusion**:  
XGBoost with cost-sensitive learning and threshold tuning delivered the **best overall performance**.

---

## 4. Final Best Model Selection

### 🏆 **Final Selected Model: XGBoost (CPU Optimized)**

**Why XGBoost was selected:**
- Highest **Precision-Recall AUC**
- Best balance between fraud detection and false positives
- No dependency on SMOTE
- Stable and scalable on CPU
- Highly explainable using SHAP values

---

## 5. Effectiveness in Catching Fraud

From the final confusion matrix:

- **Frauds correctly detected**: 79 out of 98
- **Frauds missed**: 19
- **False alarms (legitimate flagged as fraud)**: Only 5 out of ~57,000

This demonstrates:
- Strong fraud detection capability
- Minimal impact on genuine customers
- Practical usability in real-world banking systems

---

## 6. Business Value of the Final Model

### 💼 Operational Impact
- **Reduced financial losses** by catching ~80% of fraud cases
- **Minimal customer friction** due to very low false positives
- Enables banks to focus investigations on high-risk transactions

### 📊 Strategic Value
- Threshold can be adjusted dynamically based on business risk appetite
- PR-AUC based optimization aligns with real-world fraud objectives
- Explainability (SHAP) supports regulatory and audit requirements

### 🔍 Regulatory & Trust Benefits
- Transparent decision-making using SHAP explanations
- Meets compliance needs for financial institutions
- Builds trust with customers and stakeholders

---

## 7. Key Learnings from the Project

- Accuracy is misleading for imbalanced problems
- SMOTE helps **linear models**, but often **hurts tree-based models**
- Cost-sensitive learning is more effective than oversampling
- Threshold optimization is critical for fraud detection
- PR-AUC is the most meaningful evaluation metric
- Explainability is essential for real-world deployment

---

## 8. Final Conclusion

This project successfully demonstrates that **XGBoost with cost-sensitive learning and threshold optimization** is the most effective approach for credit card fraud detection on highly imbalanced data.

The final solution achieves:
- High fraud detection capability
- Low false positive rate
- Strong business and operational value
- Explainable, scalable, and production-ready performance

This makes the model suitable for **real-world financial fraud detection systems**.

---


