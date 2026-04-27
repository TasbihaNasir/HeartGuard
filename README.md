# HeartGuard 🫀
**A Single-Agent AI System for Heart Disease Risk Prediction using Artificial Neural Networks**


---

## Overview
HeartGuard is a binary classification system that predicts whether a patient is at risk of heart disease based on 13 clinical features. A trained ANN is wrapped in a single inference agent that accepts raw patient data and returns a risk assessment.

---

## Dataset
**UCI Heart Disease Dataset (Cleveland subset)**
- 303 patient records → 297 after cleaning
- 13 features: age, sex, chest pain type, blood pressure, cholesterol, ECG results, max heart rate, and more
- Target: `0` = No Disease, `1` = Disease present
- Source: https://archive.ics.uci.edu/ml/datasets/heart+disease

---

## Pipeline
```
Raw Data → EDA → Preprocessing → ANN Training → Evaluation → Inference Agent
```

1. **EDA** — correlation analysis, boxplots, class balance check
2. **Preprocessing** — missing value removal, one-hot encoding (cp, restecg, slope, thal), StandardScaler
3. **Model** — Sequential ANN: `Input(22) → Dense(64, ReLU) → Dropout(0.3) → Dense(32, ReLU) → Dropout(0.3) → Dense(1, Sigmoid)`
4. **Training** — Adam optimizer, binary crossentropy loss, EarlyStopping (patience=15)
5. **Evaluation** — 85% accuracy, 91% precision, 75% recall on held-out test set

---

## Results
| Metric | Score |
|--------|-------|
| Accuracy | 85% |
| Precision (Disease) | 0.91 |
| Recall (Disease) | 0.75 |
| F1-Score (Disease) | 0.82 |
| True Negatives | 30 |
| True Positives | 21 |
| False Negatives | 7 |

---

## Usage
```python
result = heartguard_predict({
    'age': 63, 'sex': 1, 'cp': 3, 'trestbps': 145,
    'chol': 233, 'fbs': 1, 'restecg': 0, 'thalach': 150,
    'exang': 0, 'oldpeak': 2.3, 'slope': 0, 'ca': 0, 'thal': 1
})
# {'prediction': 1, 'probability': 0.82, 'risk': 'HIGH RISK'}
```

---

## Tech Stack
- Python, Google Colab
- TensorFlow / Keras
- Scikit-learn, Pandas, NumPy
- Matplotlib, Seaborn

---
