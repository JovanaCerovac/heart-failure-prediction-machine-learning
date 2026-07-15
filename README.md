# heart-failure-prediction-machine-learning
Machine learning project for predicting heart failure events using clinical data and ensemble classification models.

## Project Overview

This project focuses on predicting heart failure events using machine learning classification algorithms.

The goal is to develop and compare different machine learning models capable of predicting patient mortality based on clinical records.

The project includes data preprocessing, exploratory data analysis, handling class imbalance, model optimization and evaluation.

---

## Dataset

The dataset contains clinical records of patients with heart failure.

The target variable is:

- `DEATH_EVENT` - indicates whether a patient experienced a fatal event.

Features include:

- Age
- Anaemia
- Diabetes
- High blood pressure
- Serum creatinine
- Serum sodium
- Ejection fraction
- Platelets
- Smoking status
- Sex
- Creatinine phosphokinase

---

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn
- Matplotlib
- Seaborn

---

# Methodology

The machine learning pipeline consists of several steps:

## 1. Data Preprocessing

Performed operations:

- Loading dataset
- Checking missing values
- Statistical analysis
- Handling missing values using SimpleImputer
- Data type conversion


## 2. Exploratory Data Analysis (EDA)

The dataset was analyzed using:

- Histograms
- Correlation matrix
- Pair plots
- Distribution analysis of binary features


## 3. Data Balancing

The dataset contains imbalanced classes.

SMOTE (Synthetic Minority Oversampling Technique) was applied to balance the training dataset.

---

## 4. Machine Learning Models

Several classification algorithms were implemented:

### Stacking Classifier

Combination of:

- Random Forest
- Gradient Boosting

Final estimator:

- Logistic Regression


### Bagging Classifier

Ensemble model based on multiple Logistic Regression estimators.


### Boosting Classifier

Gradient Boosting algorithm was used for classification.

---

## 5. Hyperparameter Optimization

Grid Search Cross Validation was used to find optimal parameters.

5-fold cross-validation was applied for model evaluation.

---

# Model Evaluation

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC


ROC curves were generated to compare model performance.

---

# Feature Selection

Recursive Feature Elimination (RFE) was applied to identify the most important features.

The best five features were selected and evaluated.

---

# Results

Results and generated plots can be found in the `results` folder.

Examples:

- Correlation matrix
- ROC curves
- Model comparison
  
---

# Project Structure

src/
heart_failure_prediction.py

data/
heart_failure_clinical_records_dataset.csv

results/
generated plots


---

# Conclusion

This project demonstrates a complete machine learning workflow for medical data classification, including preprocessing, exploratory analysis, balancing, model training, optimization and evaluation.

Ensemble learning methods were compared to determine the most suitable approach for predicting heart failure events.
