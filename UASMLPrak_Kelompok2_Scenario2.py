import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)

# 1) Load and preprocess the dataset
# -----------------------------------
df = pd.read_excel('BlaBla.xlsx')

# Rename 'UMUR_TAHUN' to 'Umur' if present
if 'UMUR_TAHUN' in df.columns:
    df = df.rename(columns={'UMUR_TAHUN': 'Umur'})

# Drop column 'A' if it exists
if 'A' in df.columns:
    df = df.drop(columns=['A'])

# Cek missing values
print("\nJumlah missing values per kolom:")
print(df.isnull().sum())

# Checking Outliers and replacing them with mean
def replace_outliers_with_mean(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    mean_value = data[column].mean()
    data[column] = np.where((data[column] < lower_bound) | (data[column] > upper_bound), mean_value, data[column])
    return data

numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    outlier_count = ((df[col] < df[col].quantile(0.25) - 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25))) |
                     (df[col] > df[col].quantile(0.75) + 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25)))).sum()
    print(f"Jumlah outlier di kolom {col}: {outlier_count}")
    df = replace_outliers_with_mean(df, col)

# Convert 'Umur' to numeric and drop invalid rows
df['Umur'] = pd.to_numeric(df['Umur'], errors='coerce')
df = df.dropna(subset=['Umur']).reset_index(drop=True)
df_clean = df.dropna().reset_index(drop=True)

# 2) Encode 'Umur' into 5 bins
# ----------------------------
bins = [0, 20, 30, 40, 50, np.inf]
labels = [1, 2, 3, 4, 5]
df_clean['Umur_binned'] = pd.cut(df_clean['Umur'], bins=bins, labels=labels)
df_clean = df_clean.drop(columns=['Umur'])

# 3) Split into features (X) and target (y)
# -----------------------------------------
X = df_clean.drop(columns=['N'])
y = df_clean['N']

# Pastikan target memiliki lebih dari 1 kelas
print("Distribusi label target:")
print(y.value_counts())
if len(y.unique()) < 2:
    raise ValueError("Target hanya memiliki satu kelas. Klasifikasi tidak bisa dilakukan.")

# Label encode umur binned
le = LabelEncoder()
X['Umur_binned'] = le.fit_transform(X['Umur_binned'])

# 4) Chi-square feature selection
# -------------------------------
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

selector = SelectKBest(score_func=chi2, k='all')
selector.fit(X_scaled, y)
chi2_scores = selector.scores_

chi2_df = pd.DataFrame({
    'Feature': X.columns,
    'Chi2 Score': chi2_scores
}).sort_values(by='Chi2 Score', ascending=False)

top5 = chi2_df['Feature'].iloc[:5].tolist()
X_selected = X[top5]

# 5) Train-test split
# -------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y
)

# 6) Define models
# ----------------
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'LightGBM': LGBMClassifier(random_state=42)
}

results = []

# 7) Train & evaluate
# -------------------
for name, model in models.items():
    if name == 'SVM':
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        proba = model.predict_proba(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        proba = model.predict_proba(X_test)

    # Prevent Errors When Only One Class is Present
    y_proba = proba[:, 1] if proba.shape[1] > 1 else np.zeros_like(y_pred)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n===== {name} =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"AUC-ROC  : {auc:.4f}")

    results.append({
        'Model': name,
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1-Score': round(f1, 4),
        'AUC-ROC': round(auc, 4)
    })

    # Visualization per model
    plt.figure(figsize=(8, 6))
    if name == 'Decision Tree':
        plot_tree(
            model,
            feature_names=top5,
            class_names=[str(c) for c in model.classes_],
            filled=True, rounded=True, fontsize=8
        )
        plt.title(f"{name} Structure")

    elif name in ['Random Forest', 'XGBoost', 'LightGBM']:
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.bar(range(len(top5)), importances[indices], align='center')
        plt.xticks(range(len(top5)), [top5[i] for i in indices], rotation=45)
        plt.title(f"{name} Feature Importances")
        plt.ylabel("Importance Score")

    elif name == 'SVM':
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{name} ROC Curve")
        plt.legend(loc="lower right")

    plt.tight_layout()
    plt.show()

# 8) Summary Table
results_df = pd.DataFrame(results).set_index('Model')

plt.figure(figsize=(6, 4))
plt.axis('off')
table = plt.table(
    cellText=results_df.values,
    colLabels=results_df.columns,
    rowLabels=results_df.index,
    loc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.2)
plt.title("Summary Table: Model Performance")
plt.show()
