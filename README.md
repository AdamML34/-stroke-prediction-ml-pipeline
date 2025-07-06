# Stroke Prediction ML Pipeline 🧠💉

This project builds a complete machine learning pipeline to predict the likelihood of stroke from patient healthcare data. It handles everything from cleaning and visualization to model training and deployment-ready model saving.

---

## 🔍 Dataset

- **Source**: [Stroke Prediction Dataset on Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- **Records**: 5,110
- **Features**: Age, gender, BMI, glucose levels, hypertension, heart disease, smoking status, and more.

---

## 🎯 Project Objectives

- Clean and preprocess real-world healthcare data
- Explore feature relationships and stroke imbalance
- Apply **SMOTE** to handle class imbalance
- Compare **Logistic Regression** vs **Random Forest**
- Save both models using `joblib` for deployment

---

## 📊 Visualizations (EDA)

- Countplots for stroke and smoking status
- Correlation heatmap
- Class balance barplot

---

## ⚙️ Preprocessing Steps

- Dropped non-informative columns (ID)
- Filled missing `bmi` with median
- Encoded categorical columns using `get_dummies`
- Split data into training and validation sets
- Applied `SMOTE` **only on training data** to balance stroke cases

---

## 🤖 Models

### 1. Logistic Regression
- Simple, interpretable
- Applied with L2 regularization

### 2. Random Forest
- Ensemble-based
- Tuned using `max_depth`, `min_samples_leaf`, and `ccp_alpha`

---

## 📈 Evaluation Metrics

- **Classification report** (precision, recall, f1-score)
- **Confusion matrix**
- **ROC AUC score**

### Final Results (Validation):

| Model             | Stroke Recall | Stroke Precision | AUC Score |
|------------------|----------------|------------------|-----------|
| LogisticRegression | 0.00         | 0.00             | 0.84      |
| RandomForest       | 0.56         | 0.17             | 0.83      |

> 📌 Random Forest achieved significantly higher recall on stroke cases, which is critical in medical diagnosis.

---

## 🧪 Tools & Libraries Used

- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `scikit-learn`, `imblearn`
- `joblib`

---

## 💾 Output

- Two trained models saved:
  - `model1`: Logistic Regression
  - `model2`: Random Forest Classifier

---

## 📁 How to Run

```bash
pip install -r requirements.txt
python stroke_pipeline.py
