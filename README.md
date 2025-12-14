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

## 📈 Model Performance (Best Model – XGBoost with Hyperparameter Tuning)
- Accuracy: ~99.9%
- Precision (Fraud): ~94%
- Recall (Fraud): ~81%
- F1-score (Fraud): ~87%
- Very low false positives and false negatives

---

## 🔍 Explainability (SHAP)
- Identified top fraud-driving features:
  - `V14`, `V4`, `V12`, `V10`, `V8`
- SHAP plots provide both global and local interpretability
- Enables trust, auditability, and regulatory compliance

---

## ✅ Key Takeaways
- Accuracy alone is misleading for imbalanced datasets
- Recall is critical in fraud detection
- Explainable AI is essential in financial applications
- The model balances business risk and detection effectiveness

---

## 🚀 Future Enhancements
- Threshold optimization
- Real-time fraud detection pipeline
- Model monitoring and retraining
- Integration with streaming data (Kafka / APIs)

---

## 📌 Author
Capstone Project – Credit Card Fraud Detection

