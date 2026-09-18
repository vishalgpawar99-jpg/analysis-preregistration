import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

# --- 1. Synthetic Dataset Generation & Preprocessing ---
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'recency_days': np.random.exponential(scale=30, size=n_samples),
    'frequency_orders': np.random.poisson(lam=5, size=n_samples),
    'monetary_value': np.random.gamma(shape=2, scale=100, size=n_samples),
    'tenure_months': np.random.randint(1, 48, size=n_samples),
    'support_tickets': np.random.poisson(lam=1.5, size=n_samples),
    'discount_usage_pct': np.random.uniform(0, 0.5, size=n_samples),
})

# Define target variable (1 = Churned, 0 = Retained) based on logical rule + noise
logit = (
    0.03 * data['recency_days']
    - 0.4 * data['frequency_orders']
    - 0.005 * data['monetary_value']
    + 0.5 * data['support_tickets']
    - 0.05 * data['tenure_months']
)
probabilities = 1 / (1 + np.exp(-logit))
data['churn'] = (probabilities > 0.45).astype(int)

# --- 2. Train / Test Split & Feature Scaling ---
X = data.drop(columns=['churn'])
y = data['churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 3. Model Training & Evaluation ---
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
}

results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": auc
    })

results_df = pd.DataFrame(results)
print("Model Comparison Summary:")
print(results_df.to_string(index=False))

# --- 4. ROC Curve Visualization ---
plt.figure(figsize=(8, 6))
for name, model in models.items():
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_test, y_proba):.2f})")

plt.plot([0, 1], [0, 1], 'k--', label="Random Baseline")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Model Performance Comparison")
plt.legend()
plt.grid(True)
plt.show()

# --- 5. Feature Importance Analysis ---
gb_model = models["Gradient Boosting"]
importances = gb_model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(8, 5))
sns.barplot(x=importances, y=feature_names, palette="viridis")
plt.title("Feature Importance Analysis (Gradient Boosting)")
plt.xlabel("Relative Importance")
plt.ylabel("Features")
plt.show()
