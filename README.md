# Credit Card Fraud Detection System

## Project Overview
This project focuses on building a **fraud detection pipeline** using the popular Kaggle [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). Fraud detection is a core challenge in fintech, where the cost of missed fraud is very high, but excessive false alarms can damage customer trust.  

The dataset contains **284,807 transactions** with only **492 fraud cases (0.17%)**, making it highly imbalanced.  

The goal: **develop models that maximize fraud detection (recall) while minimizing false alarms (precision)**.  

---

##  Workflow
1. **Data Preprocessing**
   - Scaled raw features `Time` and `Amount` using `StandardScaler`.  
   - Kept anonymized PCA features `V1–V28` as-is.  
   - Stratified train-test split (80/20).  

2. **Class Imbalance Handling**
   - Tested two strategies:
     - **SMOTE oversampling** (balanced training data).
     - **Decision threshold tuning** (optimized cutoff probability).  

3. **Models Tested**
   - Logistic Regression   
   - Random Forest   
   - XGBoost   

4. **Evaluation Metrics**
   - Precision, Recall, F1-score (for fraud class).  
   - ROC AUC and PR AUC (better for imbalanced data).  
   - Confusion matrices to understand false positives/negatives.  

---

## Results

| Model                   | Precision (Fraud) | Recall (Fraud) | F1 (Fraud) | PR AUC | Notes |
|--------------------------|------------------|----------------|------------|--------|-------|
| Logistic Regression      | 6%               | 92%            | 0.11       | 0.72   | High recall, unusable precision |
| Logistic + SMOTE         | 5.8%             | 92%            | 0.11       | 0.72   | No gain from SMOTE |
| RF + SMOTE               | 85.4%            | 83.7%          | 0.845      | 0.877  | Excellent trade-off |
| XGBoost (0.5)            | 84.4%            | 82.7%          | 0.835      | 0.877  | Matches RF+SMOTE |
| **XGBoost (tuned ~0.93)**| **90.9%**        | **81.6%**      | **0.860**  | **0.877** | 🔥 Best balance |

---

## Key Insights
- **Class imbalance handling matters**: SMOTE improved Random Forest but not Logistic Regression.  
- **Threshold tuning is powerful**: adjusting cutoff improved F1 and reduced false positives.  
- **XGBoost was the best performer**:  
  - Achieved **91% precision** and **82% recall**.  
  - Only **8 false positives** out of ~57k legit transactions.  
  - Excellent balance between catching fraud and minimizing customer disruption.  

---

## Business Impact
At the tuned threshold, the system:  
- Detected **80+ fraudulent transactions** out of 98 in the test set.  
- Reduced **false alarms by >99% compared to baseline Logistic Regression**.  
- Could save thousands of dollars in prevented fraud losses with minimal impact on legitimate users.  

---

##  Tech Stack
- Python, Pandas, NumPy  
- scikit-learn, imbalanced-learn  
- XGBoost, SHAP (explainability)  
- Matplotlib, Seaborn  


