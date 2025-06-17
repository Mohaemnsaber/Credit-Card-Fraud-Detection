# Credit Card Fraud Detection

This project applies machine learning techniques to detect fraudulent credit card transactions using a publicly available dataset with 284,808 records and 31 features.

## 📊 Dataset Description

The dataset includes:
- **28 anonymized features** (`V1` to `V28`) from PCA.
- **Time** and **Amount** (scaled in preprocessing).
- **Class** (target):  
  - `0`: Genuine transaction  
  - `1`: Fraudulent transaction

## 🔍 Project Workflow

1. **Data Cleaning & Preprocessing**
   - Checked for nulls and missing values.
   - Scaled `Amount` using `StandardScaler`.
   - Removed original `Time` column due to low correlation.

2. **Exploratory Data Analysis (EDA)**
   - Plotted distribution of `Amount` and `Time`.
   - Visualized correlation between features and fraud.

3. **Handling Class Imbalance**
   - Used **SMOTE (Synthetic Minority Oversampling Technique)** to oversample fraud cases.

4. **Modeling**
   - Trained a **Random Forest Classifier** on the balanced dataset.
   - Split data into train/test sets (70%/30%).

5. **Evaluation Metrics**
   - Achieved near-perfect precision, recall, and F1-score:
     ```
     Accuracy:     99.99%
     Precision:    0.9998 (fraud class)
     Recall:       1.0000 (fraud class)
     F1-score:     0.9999
     ```
     ## 📁 Dataset Source

The dataset used in this project is available on [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

To run this notebook:
1. Download `creditcard.csv` from the link above.
2. Place it in the root project folder before running the notebook.

## 📦 Requirements

Install the dependencies with:

```bash
pip install -r requirements.txt
