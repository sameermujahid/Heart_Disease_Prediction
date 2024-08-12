import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv('C:\\Users\\samee\\PycharmProjects\\Heart_Disease_Prediction_Project\\Heart_Disease_Prediction.csv')

# Data Cleaning

# Checking for missing values
print("Missing values in each column:\n", data.isnull().sum())
for col in data.columns:
    if data[col].isnull().sum() == 0:
        print(f'No missing values in {col} -> {data[col].isnull().sum()}')

# Identify categorical and numerical columns
cat_columns = []
num_columns = []
for col in data.columns:
    if data[col].dtype == 'object':
        cat_columns.append(col)
    else:
        num_columns.append(col)
print("Categorical columns:", cat_columns)
print("Numerical columns:", num_columns)

# Convert categorical column to numerical format
data['Heart Disease'] = data['Heart Disease'].map({'Presence': 1, 'Absence': 0})

# Outlier Detection and Removal
# Trimming based on 5th and 95th percentiles
cols_to_trim = ['Chest pain type', 'Cholesterol', 'BP', 'FBS over 120', 'Max HR', 'ST depression', 'Number of vessels fluro']

for col in cols_to_trim:
    lower_limit = data[col].quantile(0.05)
    upper_limit = data[col].quantile(0.95)
    data[col] = data[col].apply(lambda x: x if lower_limit <= x <= upper_limit else np.nan)

data = data.dropna(subset=cols_to_trim)

print("\nSample data after trimming:")
print(data[cols_to_trim].head())
print(data.head())

# Plot boxplots for numerical columns
plt.figure(figsize=(12, 8))
for i, col in enumerate(data.columns, 1):
    plt.subplot(len(data.columns) // 3 + 1, 3, i)
    sns.boxplot(y=data[col])
    plt.xlabel('Value')
    plt.ylabel(col)
    plt.title(col)
    plt.tight_layout()
plt.show()

# Prepare features and target variable
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Training

# Logistic Regression
log_reg_model = LogisticRegression(random_state=42)
log_reg_model.fit(X_train_scaled, y_train)
log_reg_preds = log_reg_model.predict(X_test_scaled)

# Decision Tree
dec_tree_model = DecisionTreeClassifier(random_state=42)
dec_tree_model.fit(X_train_scaled, y_train)
dec_tree_preds = dec_tree_model.predict(X_test_scaled)

# Random Forest
rand_forest_model = RandomForestClassifier(random_state=42)
rand_forest_model.fit(X_train_scaled, y_train)
rand_forest_preds = rand_forest_model.predict(X_test_scaled)

# Support Vector Machine (SVM)
svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)
svm_preds = svm_model.predict(X_test_scaled)

# Model Evaluation
models = {
    "Logistic Regression": log_reg_preds,
    "Decision Tree": dec_tree_preds,
    "Random Forest": rand_forest_preds,
    "SVM": svm_preds
}

for model_name, preds in models.items():
    print(f"\nModel: {model_name}")
    print("Classification Report:\n", classification_report(y_test, preds))
    print("Accuracy:", accuracy_score(y_test, preds))
    auc_score = roc_auc_score(y_test, preds)
    fpr, tpr, _ = roc_curve(y_test, preds)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='best')
    plt.show()

    # Display Confusion Matrix
    conf_matrix = confusion_matrix(y_test, preds)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()

# Hyperparameter Tuning for Logistic Regression
param_grid_log_reg = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga']
}

grid_search_log_reg = GridSearchCV(estimator=log_reg_model, param_grid=param_grid_log_reg, cv=5, n_jobs=-1, verbose=2)
grid_search_log_reg.fit(X_train_scaled, y_train)

print("Best parameters for Logistic Regression:", grid_search_log_reg.best_params_)

best_log_reg_model = grid_search_log_reg.best_estimator_

# Evaluate the tuned Logistic Regression model
best_log_reg_preds = best_log_reg_model.predict(X_test_scaled)
print("\nTuned Logistic Regression Model:")
print("Classification Report:\n", classification_report(y_test, best_log_reg_preds))
print("Accuracy:", accuracy_score(y_test, best_log_reg_preds))

# Save the tuned Logistic Regression model
joblib.dump(best_log_reg_model, 'logistic_regression_model.pkl')
