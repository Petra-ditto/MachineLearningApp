"""
Loan Eligibility Prediction - End-to-end script

Covers:
1) Imports and data loading
2) EDA (basic statistics, missing values)
3) Preprocessing and train/test split
4) Logistic Regression (training, evaluation, coefficients)
5) Random Forest (training, evaluation, feature importance)
6) Utility: plots, save models

"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


##########################
# 1. IMPORT LIBRARIES & DATA
##########################

DATA_PATH = "/Users/petraszabolcsi/Documents/Excel/Loan Eligibility Prediction.csv"
MODEL_DIR = "/Users/petraszabolcsi/Documents/Excel/Models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)


##########################
# 2. EXPLORATORY DATA ANALYSIS
##########################

def eda_report(df: pd.DataFrame, show_plots: bool = True):
    print("== BASIC INFO ==")
    print(df.info())
    print("\n== SHAPE ==")
    print(df.shape)

    print("\n== FIRST 5 ROWS ==")
    print(df.head())

    print("\n== SUMMARY (NUMERIC) ==")
    print(df.describe())

    print("\n== SUMMARY (ALL) ==")
    print(df.describe(include='all'))

    print("\n== MISSING VALUES ==")
    print(df.isnull().sum())

    # Basic distribution of the target
    if 'Loan_Status' in df.columns:
        print("\n== TARGET VALUE COUNTS ==")
        print(df['Loan_Status'].value_counts(dropna=False))

    if show_plots:
        # Target histogram
        if 'Loan_Status' in df.columns:
            fig, ax = plt.subplots(figsize=(6,4))
            df['Loan_Status'].map({'Y':1,'N':0}).value_counts().plot(kind='bar', ax=ax)
            ax.set_title('Loan_Status distribution (1=Y,0=N)')
            ax.set_xticklabels(['Rejected (0)','Approved (1)'])
            plt.tight_layout()
            plt.show()

        # Numeric histograms
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            df[num_cols].hist(bins=15, figsize=(12, 8))
            plt.tight_layout()
            plt.show()


# Run EDA - if this line is enabled, it runs EDA but then doesn't run the rest
#eda_report(df)


##########################
# 3. PREPROCESSING & TRAIN/TEST SPLIT
##########################

# Copy original so we don't mutate user's df
data = df.copy()

# Standardize column names (optional)
# Map target to numeric
if 'Loan_Status' in data.columns:
    data['Loan_Status'] = data['Loan_Status'].map({'Y': 1, 'N': 0})

# If Dependents encoded as '3+' (sometimes in such datasets), normalize it
if data['Dependents'].dtype == object:
    data['Dependents'] = data['Dependents'].replace('3+', 3).astype(float)

print('I made it until here')

# Fill missing values with sensible defaults or indicators
# We'll handle numeric missing values with median and categorical missing with 'Missing'
numeric_cols = ['Applicant_Income', 'Coapplicant_Income', 'Loan_Amount', 'Loan_Amount_Term', 'Credit_History']
# Ensure columns exist (robustness)
numeric_cols = [c for c in numeric_cols if c in data.columns]
cat_cols = [c for c in ['Gender','Married','Dependents','Education','Self_Employed','Property_Area'] if c in data.columns]

# Simple imputation (we'll integrate into pipeline) - but let's show missing count
print('\nNumeric missing before imputation:')
print(data[numeric_cols].isnull().sum())
print('\nCategorical missing before imputation:')
print(data[cat_cols].isnull().sum())

# Prepare X and y
FEATURE_COLS = cat_cols + numeric_cols
X = data[FEATURE_COLS]
y = data['Loan_Status'] if 'Loan_Status' in data.columns else None

# Train/test split
if y is None:
    raise ValueError('Target column Loan_Status not found in dataset. Please ensure it exists.')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#test_size means 20% of data will be used for testing. If not specified, the default is 25%.
#random_state: controls the shuffling applied to the data before the split. Passing an integer (like 42) ensures that your code outputs the exact same split every time it is run.
#stratify: used for classification tasks. If you pass your target array here (e.g., stratify=y), the split will maintain the same proportion of class labels in both the training and test sets as the original dataset.

print('\nTrain/Test sizes:', X_train.shape, X_test.shape)

# Preprocessing: categorical OneHot + numeric passthrough (could scale numeric if using Logistic Regression)
# OneHotEncoder is a data preprocessing technique in machine learning that converts categorical data into numerical
# format that the model can use

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, cat_cols),
        ('num', numeric_transformer, numeric_cols)
    ],
    remainder='drop'
)


##########################
# 4. LOGISTIC REGRESSION
##########################

logreg_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('clf', LogisticRegression(max_iter=2000, solver='lbfgs'))
])

print('\nTraining Logistic Regression...')
logreg_pipeline.fit(X_train, y_train)

# Predictions
y_pred_lr = logreg_pipeline.predict(X_test)
y_prob_lr = logreg_pipeline.predict_proba(X_test)[:, 1]

print('\n=== Logistic Regression Evaluation ===')
print(classification_report(y_test, y_pred_lr))
print('Accuracy:', accuracy_score(y_test, y_pred_lr))
print('ROC AUC:', roc_auc_score(y_test, y_prob_lr))

# Cross-validated AUC (optional)
cv_auc = cross_val_score(logreg_pipeline, X, y, cv=5, scoring='roc_auc')
print('5-fold CV ROC AUC (LogReg):', cv_auc.mean(), '±', cv_auc.std())

# Coefficients (map back to feature names)
# Extract one-hot feature names
onehot = logreg_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
if hasattr(onehot, 'get_feature_names_out'):
    ohe_features = onehot.get_feature_names_out(cat_cols).tolist()
else:
    # fallback
    ohe_features = []

feature_names = ohe_features + numeric_cols
coeffs = logreg_pipeline.named_steps['clf'].coef_[0]

coef_df = pd.DataFrame({'feature': feature_names, 'coefficient': coeffs})
coef_df = coef_df.reindex(coef_df.coefficient.abs().sort_values(ascending=False).index)
print('\nTop logistic regression coefficients:')
print(coef_df.head(20))


##########################
# 5. RANDOM FOREST
##########################

rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
])

print('\nTraining Random Forest...')
rf_pipeline.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_pipeline.predict(X_test)
y_prob_rf = rf_pipeline.predict_proba(X_test)[:, 1]

print('\n=== Random Forest Evaluation ===')
print(classification_report(y_test, y_pred_rf))
print('Accuracy:', accuracy_score(y_test, y_pred_rf))
print('ROC AUC:', roc_auc_score(y_test, y_prob_rf))

cv_auc_rf = cross_val_score(rf_pipeline, X, y, cv=5, scoring='roc_auc')
print('5-fold CV ROC AUC (RF):', cv_auc_rf.mean(), '±', cv_auc_rf.std())

# Feature importance for Random Forest
# We need feature names after preprocessing (including one-hot)
# Create preprocessed training matrix to extract feature names
preprocessor_fit = preprocessor.fit(X_train)
X_train_transformed = preprocessor_fit.transform(X_train)

# Build feature names list
try:
    ohe = preprocessor_fit.named_transformers_['cat'].named_steps['onehot']
    ohe_names = ohe.get_feature_names_out(cat_cols).tolist()
except Exception:
    ohe_names = []

all_feature_names = ohe_names + numeric_cols

# Extract feature importances from the underlying RF (in pipeline)
rf_model = rf_pipeline.named_steps['clf']
importances = rf_model.feature_importances_
fi_df = pd.DataFrame({'feature': all_feature_names, 'importance': importances})
fi_df = fi_df.sort_values('importance', ascending=False)

print('\nTop features by Random Forest importance:')
print(fi_df.head(20))

# Plot feature importance (top 15)
fig, ax = plt.subplots(figsize=(8,6))
fi_df.head(15).plot.barh(x='feature', y='importance', ax=ax, legend=False)
ax.invert_yaxis()
ax.set_title('Random Forest - Top 15 Feature Importances')
plt.tight_layout()
plt.show()


##########################
# 6. CONFUSION MATRICES
##########################

fig, axes = plt.subplots(1, 2, figsize=(12,5))
cm_lr = confusion_matrix(y_test, y_pred_lr)
cm_rf = confusion_matrix(y_test, y_pred_rf)

disp_lr = ConfusionMatrixDisplay(confusion_matrix=cm_lr, display_labels=['Rejected','Approved'])
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=['Rejected','Approved'])

disp_lr.plot(ax=axes[0])
axes[0].set_title('Logistic Regression')

disp_rf.plot(ax=axes[1])
axes[1].set_title('Random Forest')

plt.tight_layout()
plt.show()


##########################
# 7. SAVE MODELS + ARTIFACTS
##########################

logreg_path = os.path.join(MODEL_DIR, 'logreg_pipeline.joblib')
rf_path = os.path.join(MODEL_DIR, 'rf_pipeline.joblib')

joblib.dump(logreg_pipeline, logreg_path)
joblib.dump(rf_pipeline, rf_path)

print('\nSaved Logistic Regression pipeline to', logreg_path)
print('Saved Random Forest pipeline to', rf_path)

# Also save coefficient and feature importance CSVs
coef_df.to_csv(os.path.join(MODEL_DIR, 'logreg_coefficients.csv'), index=False)
fi_df.to_csv(os.path.join(MODEL_DIR, 'rf_feature_importances.csv'), index=False)

print('Saved coefficient and feature importance CSVs.')


##########################
# 8. PREDICT FUNCTION FOR NEW DATA
##########################

def predict_new_application(new_df: pd.DataFrame, model_pipeline):
    """Accepts a dataframe with the same feature columns and returns prediction and probability."""
    preds = model_pipeline.predict(new_df)
    probs = model_pipeline.predict_proba(new_df)[:, 1]
    return preds, probs

# Example usage (uncomment to test with a single row)
# sample = X_test.iloc[[0]]
# print('Sample true label:', y_test.iloc[0])
# print('LogReg pred/prob:', predict_new_application(sample, logreg_pipeline))
# print('RF pred/prob:', predict_new_application(sample, rf_pipeline))

print('\nScript finished successfully.')
