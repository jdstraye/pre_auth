"""
This code is for step 2 in the plan to 
find the best hyperparameters for the model
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from scipy.stats import randint, uniform
from CustomLogger import CustomRotatingFileHandler
import logging

# --- Logging Setup ---
Path("logs").mkdir(exist_ok=True)
handler = CustomRotatingFileHandler("logs/pre_auth_train", maxBytes=10*1024*1024, backupCount=5)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)
logger.info("Logger initialized with rotating file handler.")

# --- Helper Functions (from previous code) ---
def preprocess_data(df, feature_cols, categorical_cols=None):
    """
    Preprocess data: handle categorical variables and missing values.
    """
    df = df.copy()
    if categorical_cols:
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                logger.info(f"Encoded categorical column: {col}")
    for col in feature_cols:
        if col in df.columns:
            if df[col].dtype in [np.float64, np.int64]:
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
    return df

# --- Main Execution Block ---
try:
    data_path = Path("data/splits")
    train_path = data_path / "train_latest.csv"
    test_path = data_path / "test_latest.csv"
    train_file = os.path.realpath(train_path)
    test_file = os.path.realpath(test_path)
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    logger.info(f"Loaded and split data. Training set size: {len(train_df)}, Test set size: {len(test_df)}")
except FileNotFoundError as e:
    logger.error(f"Error loading data: {e}. Please ensure {test_file} and {train_file} exist.")
    raise

feature_cols = [
    "user_initials", "DebtToIncome", "Automatic Financing_Score",
    "Automatic Financing_below_600?", "Automatic Financing_Status_Approved?",
    "Automatic Financing_Status_Declined?", "Automatic Financing_Amount",
    "Automatic Financing_Details_in_the_wallet?", "Automatic Financing_Details_just_available?",
    "Automatic Financing_DebtToIncome", "0% Unsecured Funding_missing?",
    "0% Unsecured Funding_Score", "0% Unsecured Funding_below_600?",
    "Unsecured Funding_Status_As_is?", "Unsecured Funding_Status_Declined?",
    "Unsecured Funding_Status_if_Fixed?", "Unsecured Funding_Status_NA?",
    "0% Unsecured Funding_Amount", "0% Unsecured Funding_Details_To_book?",
    "0% Unsecured Funding_Details_just_CL?", "0% Unsecured Funding_Details_NA?",
    "0% Unsecured Funding_Collections", "0% Unsecured Funding_Contingencies",
    "0% Unsecured Funding_DebtToIncome", "Debt Resolution_missing?",
    "Debt Resolution_Score", "Debt Resolution_score_missing?",
    "Debt Resolution_below_600?", "Debt Resolution_Status_Approved?",
    "Debt Resolution_Status_Declined?", "Debt Resolution_Status_NA?",
    "Debt Resolution_Amount", "Debt Resolution_DebtToIncome",
]
categorical_cols = ["user_initials"]

logger.debug(train_df.columns)
missing_cols = [col for col in feature_cols if col not in train_df.columns]
if missing_cols:
    logger.debug(f"Missing columns: {missing_cols}")
try:
    train_df = preprocess_data(train_df, feature_cols, categorical_cols)
    test_df = preprocess_data(test_df, feature_cols, categorical_cols)
except Exception as e:
    logger.error(f"Error preprocessing data: {e}")
    raise

# --- Step 2: Multi-Algorithm Optimization with RandomizedSearchCV ---
logger.info("\n--- Step 2: Optimizing Multiple Models with RandomizedSearchCV ---")

pipeline = Pipeline([
    ('classifier', RandomForestClassifier())
])

# Define the hyperparameter search space
param_distributions = {
    'RandomForestClassifier': {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [None, 3, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 5, 10]
    },
    'XGBClassifier': {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [3, 5, 10],
        'learning_rate': [0.01, 0.1, 0.5],
        'gamma': [0, 0.1, 0.5]
    },
    'LGBMClassifier': {
        'n_estimators': [100, 200, 300, 400, 500],
        'num_leaves': [31, 62, 127],
        'learning_rate': [0.01, 0.1, 0.5],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0, 0.1, 0.5]
    },
    'CatBoostClassifier': {
        'iterations': [100, 200, 300, 400, 500],
        'depth': [3, 5, 10],
        'learning_rate': [0.01, 0.1, 0.5],
        'l2_leaf_reg': [1, 3, 5]
    },
    'LogisticRegression': {
        'C': [0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    },
    'SVC': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly']
    },
    'KNeighborsClassifier': {
        'n_neighbors': [3, 5, 10],
        'weights': ['uniform', 'distance']
    }
}

# Define the models
models = {
    'RandomForestClassifier': RandomForestClassifier(random_state=42),
    'XGBClassifier': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    'LGBMClassifier': LGBMClassifier(random_state=42),
    'CatBoostClassifier': CatBoostClassifier(random_state=42, verbose=0),
    'LogisticRegression': LogisticRegression(random_state=42),
    'SVC': SVC(random_state=42),
    'KNeighborsClassifier': KNeighborsClassifier()
}

def hyperparameter_search(X, y, model_name, param_dist):
    logger.info(f"Performing hyperparameter search for {model_name}...")
    model = models[model_name]
    random_search = RandomizedSearchCV(model, param_dist, cv=KFold(n_splits=5, shuffle=True, random_state=42), n_iter=20, random_state=42)
    random_search.fit(X, y)
    logger.info(f"Hyperparameter search completed for {model_name}. Best parameters: {random_search.best_params_}")
    logger.info(f"Best cross-validation accuracy score: {random_search.best_score_:.4f}")
    return random_search.best_estimator_, random_search.best_score_

def get_top_models(X, y, n_top):
    all_models = []
    total_iterations = sum(len(param_dist) for param_dist in param_distributions.values())
    current_iteration = 0
    for model_name, param_dist in param_distributions.items():
        model = models[model_name]
        #20250822a n_iter = len(param_dist)
        #20250822b n_iter = 50
        n_iter = sum(len(value) for value in param_dist.values())
        random_search = RandomizedSearchCV(model, param_dist, cv=KFold(n_splits=5, shuffle=True, random_state=42), n_iter=n_iter, random_state=42)
        random_search.fit(X, y)
        current_iteration += len(param_dist)
        logger.info(f"Completed {current_iteration} out of {total_iterations} iterations ({current_iteration/total_iterations*100:.2f}%)")
        for i in range(len(random_search.cv_results_['params'])):
            all_models.append((model_name, random_search.cv_results_['params'][i], random_search.cv_results_['mean_test_score'][i]))
    all_models.sort(key=lambda x: x[2], reverse=True)
    return all_models[:n_top]

def log_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_
        if feature_importances.ndim > 0:
            logger.info("Feature importances:")
            for feature_name, importance in zip(feature_names, feature_importances):
                logger.info(f"{feature_name}: {importance:.4f}")
        else:
            logger.info("Feature importances are not available in a format that can be logged.")
    elif hasattr(model, 'coef_'):
        feature_coefficients = model.coef_[0]
        logger.info("Feature coefficients:")
        for feature_name, coefficient in zip(feature_names, feature_coefficients):
            logger.info(f"{feature_name}: {coefficient:.4f}")
    else:
        logger.info("Feature importances not available for this model.")

def log_class_imbalance(y):
    class_counts = np.bincount(y)
    logger.info("Class imbalance:")
    for class_label, count in enumerate(class_counts):
        logger.info(f"Class {class_label}: {count} samples")

# Setup
train_path = Path(__name__).parent.parent / "data" / "splits" / "train_latest.csv"

# Train the model for phase 1 - status
train_df = pd.read_csv(train_path)
train_status_df = train_df
#20250821a X_status = train_status_df.drop("final_contract_status_label", axis=1)
X_train_status = train_status_df[feature_cols]
y_train_status = train_status_df["final_contract_status_label"]
X_train_status = preprocess_data(X_train_status, feature_cols, categorical_cols)

# Logging and Saving the top models for phase 2 - tier
top_models = get_top_models(X_train_status, y_train_status, 100)
for i, (model_name, params, score) in enumerate(top_models):
    logger.info(f"Model {i+1}: {model_name}, Params: {params}, Score: {score:.4f}")
    model = models[model_name]
    model.set_params(**params)
#20250822a    model.fit(X_train_status, y_train_status)
    log_feature_importance(model, X_train_status.columns)
    log_class_imbalance(y_train_status)
    joblib.dump(model, f"status_model_{i+1}.pkl")
    logger.info(f"Status model {i+1} saved.")

# Train the model for phase 2 - tier
train_approved_df = train_df[train_df["final_contract_status_label"] == 0]
X_train_tier = train_approved_df[feature_cols]
y_train_tier = train_approved_df["final_contract_tier_label"]
X_train_tier = preprocess_data(X_train_tier, feature_cols, categorical_cols)

# Encode y_train_tier if it's categorical
if not pd.api.types.is_numeric_dtype(y_train_tier):
    le = LabelEncoder()
    y_train_tier = le.fit_transform(y_train_tier)

logger.info(f"Training tier model...")
top_models = get_top_models(X_train_tier, y_train_tier, 100)
# Logging and Saving the top models for phase 2 - tier
for i, (model_name, params, score) in enumerate(top_models):
    logger.info(f"Model {i+1}: {model_name}, Params: {params}, Score: {score:.4f}")
    model = models[model_name]
    model.set_params(**params)
#20250822a    model.fit(X_train_tier, y_train_tier)
    log_feature_importance(model, X_train_tier.columns)
    log_class_imbalance(y_train_tier)
    joblib.dump(model, f"tier_model_{i+1}.pkl")
    logger.info(f"Tier model {i+1} saved.")
