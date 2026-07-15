"""
Heart Failure Prediction using Machine Learning

This script performs:
- data preprocessing
- exploratory data analysis
- SMOTE balancing
- ensemble model training
- model evaluation
- feature selection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE

# 1. Početno procesiranje podataka
print("1. Početno procesiranje podataka:")
df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
print(df.head())
print(df.isnull().sum())
print(df.describe())

# Popunjavanje nedostajućih vrednosti (npr. sa najčešćom vrednosti iz kolone)
imputer = SimpleImputer(strategy='most_frequent')
df[df.select_dtypes(include=[np.number]).columns] = imputer.fit_transform(df.select_dtypes(include=[np.number]))

binary_columns = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking', 'DEATH_EVENT']
for col in binary_columns:
    df[col] = df[col].astype(int)

print(df.info())
print(df.describe())

def detect_anomalies(df):
    df = df.copy()
    thresholds = {
        'age': 120,
        'creatinine_phosphokinase': 3000,
        'ejection_fraction': 80,
        'platelets': 800000,
        'serum_creatinine': 3.5,
        'serum_sodium': 160
    }
    anomaly_rows = []
    for col, threshold in thresholds.items():
        anomalies = df[col] > threshold
        df.loc[anomalies, 'Anomaly_Param'] = df.loc[anomalies, col]
        df.loc[anomalies, 'Anomaly_Param_Name'] = col
        anomaly_rows.extend(df.loc[anomalies].index.tolist())
    df_anomalies = df.loc[anomaly_rows]
    return df_anomalies

df_anomalies = detect_anomalies(df)
print("Redovi sa anomalijama, parametrom koji uzrokuje anomaliju i nazivom parametra:")
print(df_anomalies[['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium', 'Anomaly_Param', 'Anomaly_Param_Name']])

# 2. Eksplorativna analiza skupa
print("2. Eksplorativna analiza")
numerical_columns = df.select_dtypes(include=[np.number]).columns
fig, axes = plt.subplots(len(numerical_columns) // 3 + 1, 3, figsize=(15, 15))
fig.suptitle('Histogrami za numeričke atribute', fontsize=20)
for ax, col in zip(axes.flatten(), numerical_columns):
    sns.histplot(df[col], bins=40, ax=ax, kde=True, color='skyblue')
    ax.set_title(col, fontsize=10)
    ax.set_xlabel('')
    ax.set_ylabel('')
for ax in axes.flatten()[len(numerical_columns):]:
    fig.delaxes(ax)
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Korelaciona matrica', fontsize=20)
pairplot_fig = sns.pairplot(df, hue='DEATH_EVENT', vars=['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium'])
pairplot_fig.fig.suptitle('Raspodela ciljnog atributa u odnosu na nezavisne promenljive', fontsize=16)
pairplot_fig.fig.subplots_adjust(top=0.95)
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
for i, col in enumerate(binary_columns):
    row = i // 3
    col_position = i % 3
    df[col].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, ax=axs[row, col_position])
    axs[row, col_position].set_title(col)
    axs[row, col_position].set_ylabel('')
plt.tight_layout()

# 3. Balansiranje skupa podataka
print("3. Balansiranje skupa podataka")
X = df.drop('DEATH_EVENT', axis=1)
y = df['DEATH_EVENT']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Distribucija klasa pre balansiranja (Trening skup):")
print(y_train.value_counts())
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print("\nDistribucija klasa posle balansiranja (Trening skup):")
print(y_train_resampled.value_counts())

# 4. i 5. Kreiranje modela
print("4. i 5. Kreiranje modela")
def evaluate_model(model, X, y):
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    return scores.mean(), scores.std()

param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
param_grid_gb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}
param_grid_stacking = {
    'final_estimator__logistic__C': [0.1, 1.0, 10],
    'final_estimator__logistic__penalty': ['l2'],
    'rf__n_estimators': [50, 100],
    'rf__max_depth': [10, 20],
    'gb__n_estimators': [100, 200],
    'gb__learning_rate': [0.01, 0.1]
}

estimators = [
    ('rf', RandomForestClassifier(random_state=42)),
    ('gb', GradientBoostingClassifier(random_state=42))
]
stacking_model = StackingClassifier(
    estimators=estimators,
    final_estimator=Pipeline([
        ('scaler', StandardScaler()),
        ('logistic', LogisticRegression(max_iter=500, random_state=42))
    ])
)
bagging_model = BaggingClassifier(
    estimator=Pipeline([
        ('scaler', StandardScaler()),
        ('logistic', LogisticRegression(max_iter=500, random_state=42))
    ]),
    n_estimators=100,
    random_state=42
)
boosting_model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42)

models = {
    'Stacking': (stacking_model, param_grid_stacking),
    'Bagging': (bagging_model, {}),
    'Boosting': (boosting_model, param_grid_gb)
}

cv_results = {}
best_estimators = {}

for name, (model, params) in models.items():
    if params:
        grid_search = GridSearchCV(model, params, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train_resampled, y_train_resampled)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print(f"Best parameters for {name}: {best_params}")
    else:
        best_model = model
        best_model.fit(X_train_resampled, y_train_resampled)

    mean_score, std_score = evaluate_model(best_model, X_train_resampled, y_train_resampled)
    cv_results[name] = {
        'mean_accuracy': mean_score,
        'std_accuracy': std_score
    }
    best_estimators[name] = best_model

print("6. Prikaz rezultata unakrsne validacije")
for name, metrics in cv_results.items():
    print(f"{name} model CV results:")
    print(f"Mean Accuracy: {metrics['mean_accuracy']:.2f}")
    print(f"Standard Deviation of Accuracy: {metrics['std_accuracy']:.2f}")
    print("\n")

print("7. Evaluacija modela na testnom skupu i analiza rezultata")

roc_curves = {}
pr_curves = {}
model_scores = {}

for name, model in best_estimators.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    model_scores[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_curves[name] = (fpr, tpr, roc_auc)

print("Rezultati modela na testnom skupu:")
for name, scores in model_scores.items():
    print(f"{name} model scores:")
    print(f"Accuracy: {scores['accuracy']:.2f}")
    print(f"Precision: {scores['precision']:.2f}")
    print(f"Recall: {scores['recall']:.2f}")
    print(f"F1 Score: {scores['f1_score']:.2f}")
    print(f"ROC AUC: {scores['roc_auc']:.2f}")
    print("\n")

plt.figure(figsize=(10, 8))
for name, (fpr, tpr, roc_auc) in roc_curves.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

print("8. Izbor najboljih atributa")
def select_features_and_evaluate(model, X_train, y_train, X_test, y_test, n_features=5):
    rfe_selector = RFE(estimator=RandomForestClassifier(random_state=42), n_features_to_select=n_features, step=1)
    rfe_selector.fit(X_train, y_train)
    selected_features = X_train.columns[rfe_selector.support_]
    X_train_selected = rfe_selector.transform(X_train)
    X_test_selected = rfe_selector.transform(X_test)

    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_test_selected)
    y_prob = model.predict_proba(X_test_selected)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"Model scores after feature selection:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"ROC AUC: {roc_auc:.2f}")
    print(f"Selected features: {selected_features.tolist()}\n")

    return selected_features, {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }

for name, model in best_estimators.items():
    print(f"Evaluating feature selection for {name} model:")
    select_features_and_evaluate(model, X_train_resampled, y_train_resampled, X_test, y_test, n_features=5)
