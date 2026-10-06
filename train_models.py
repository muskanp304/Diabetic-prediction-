"""
Stage 2: High-Performance Supervised Machine Learning Pipeline
--------------------------------------------------------------
Trains, tunes, and benchmarks supervised classification algorithms:
1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Naive Bayes (GaussianNB)
4. Decision Tree Classifier
5. Ensemble Methods: Random Forest & Gradient Boosting & XGBoost
6. Support Vector Machine (SVM)
7. Soft-Voting Ensemble Classifier

Handles class imbalance using BorderlineSMOTE on the training split.
Optimizes hyperparameter grids with 5-Fold Stratified Cross-Validation.
Evaluates Accuracy, Precision, Recall, Specificity, F1-Score, and ROC-AUC.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from imblearn.over_sampling import BorderlineSMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

def train_and_evaluate_models(preprocessed_csv='diabetes_preprocessed.csv'):
    print("=" * 80)
    print("STAGE 2: HIGH-PERFORMANCE MODEL TRAINING & BENCHMARKING PIPELINE")
    print("=" * 80)

    # 1. Load Preprocessed Data
    print(f"\n[1] Loading preprocessed dataset from: {preprocessed_csv}")
    df = pd.read_csv(preprocessed_csv)
    
    feature_cols = [c for c in df.columns if c != 'Outcome']
    X = df[feature_cols]
    y = df['Outcome']
    print(f"    - Features: {len(feature_cols)} attributes -> {feature_cols}")
    print(f"    - Dataset Shape: {X.shape}, Target Distribution: {np.bincount(y)}")

    # 2. Stratified Train-Test Split (80/20)
    print("\n[2] Splitting into Stratified Train (80%) and Test (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"    - Train set: {len(y_train)} samples (Class 0: {np.sum(y_train==0)}, Class 1: {np.sum(y_train==1)})")
    print(f"    - Test set : {len(y_test)} samples (Class 0: {np.sum(y_test==0)}, Class 1: {np.sum(y_test==1)})")

    # 3. Feature Scaling (StandardScaler fitted strictly on training data)
    print("\n[3] Scaling features using StandardScaler (fit on train, transform test)...")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)

    os.makedirs('saved_models', exist_ok=True)
    joblib.dump(scaler, 'saved_models/scaler.joblib')
    print("    - Scaler saved to: saved_models/scaler.joblib")

    # 4. Class Imbalance Handling: BorderlineSMOTE on Training Split Only
    print("\n[4] Handling Class Imbalance with BorderlineSMOTE on Training Data:")
    bsmote = BorderlineSMOTE(random_state=42)
    X_train_res, y_train_res = bsmote.fit_resample(X_train_scaled, y_train)
    print(f"    - Before Resampling: Class 0 = {np.sum(y_train==0)}, Class 1 = {np.sum(y_train==1)}")
    print(f"    - After BorderlineSMOTE: Class 0 = {np.sum(y_train_res==0)}, Class 1 = {np.sum(y_train_res==1)}")
    print("    - [NOTE] Resampling is strictly applied to training data only to avoid data leakage.")

    # 5. Define Model Candidates & Hyperparameter Grids
    print("\n[5] Training & Cross-Validating Models...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model_configs = {
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'params': {'C': [0.1, 0.5, 1.0], 'solver': ['lbfgs', 'liblinear']}
        },
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {'n_neighbors': [5, 7, 9], 'weights': ['uniform', 'distance']}
        },
        'Naive Bayes': {
            'model': GaussianNB(),
            'params': {'var_smoothing': [1e-9, 1e-8, 1e-7]}
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {'max_depth': [3, 4, 5], 'min_samples_split': [4, 6], 'min_samples_leaf': [2, 4]}
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {'n_estimators': [150, 250], 'max_depth': [5, 6, 8], 'min_samples_leaf': [2, 3]}
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {'n_estimators': [80, 100, 140], 'learning_rate': [0.03, 0.05, 0.1], 'max_depth': [3, 4]}
        },
        'XGBoost': {
            'model': XGBClassifier(random_state=42, eval_metric='logloss'),
            'params': {'n_estimators': [80, 100, 120], 'learning_rate': [0.03, 0.05, 0.1], 'max_depth': [3, 4]}
        },
        'SVM': {
            'model': SVC(probability=True, random_state=42),
            'params': {'C': [0.8, 1.2, 2.0], 'kernel': ['rbf'], 'gamma': ['scale']}
        }
    }

    results = {}
    fitted_models = {}
    best_overall_model = None
    best_overall_score = -1.0
    best_overall_name = ""

    for name, config in model_configs.items():
        print(f"    - Optimizing {name} with 5-Fold Stratified CV...")
        grid = GridSearchCV(
            estimator=config['model'],
            param_grid=config['params'],
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1
        )
        grid.fit(X_train_res, y_train_res)
        best_est = grid.best_estimator_
        fitted_models[name] = best_est

        # Predictions on Unseen Test Set
        y_pred = best_est.predict(X_test_scaled)
        y_prob = best_est.predict_proba(X_test_scaled)[:, 1] if hasattr(best_est, 'predict_proba') else best_est.decision_function(X_test_scaled)

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred))
        auc = float(roc_auc_score(y_test, y_prob))
        cv_auc = float(grid.best_score_)

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = map(int, cm.ravel())
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        # Clinical composite score
        composite_score = (auc * 0.40) + (acc * 0.25) + (rec * 0.20) + (prec * 0.15)

        feature_imp = {}
        if hasattr(best_est, 'feature_importances_'):
            feature_imp = {f: float(imp) for f, imp in zip(feature_cols, best_est.feature_importances_)}
        elif hasattr(best_est, 'coef_'):
            coefs = np.abs(best_est.coef_[0])
            norm = coefs / np.sum(coefs) if np.sum(coefs) > 0 else coefs
            feature_imp = {f: float(c) for f, c in zip(feature_cols, norm)}

        results[name] = {
            'metrics': {
                'accuracy': round(acc, 4),
                'precision': round(prec, 4),
                'recall': round(rec, 4),
                'specificity': round(spec, 4),
                'f1_score': round(f1, 4),
                'roc_auc': round(auc, 4),
                'cv_roc_auc': round(cv_auc, 4),
                'composite_score': round(composite_score, 4)
            },
            'best_params': grid.best_params_,
            'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
            'feature_importance': feature_imp
        }

        if composite_score > best_overall_score:
            best_overall_score = composite_score
            best_overall_model = best_est
            best_overall_name = name

    # 6. Train Soft-Voting Ensemble Classifier
    print("    - Training Soft-Voting Ensemble Classifier...")
    voting_ensemble = VotingClassifier(
        estimators=[
            ('rf', fitted_models['Random Forest']),
            ('gb', fitted_models['Gradient Boosting']),
            ('xgb', fitted_models['XGBoost']),
            ('svm', fitted_models['SVM'])
        ],
        voting='soft'
    )
    voting_ensemble.fit(X_train_res, y_train_res)
    fitted_models['Voting Ensemble'] = voting_ensemble

    y_pred_v = voting_ensemble.predict(X_test_scaled)
    y_prob_v = voting_ensemble.predict_proba(X_test_scaled)[:, 1]

    acc_v = float(accuracy_score(y_test, y_pred_v))
    prec_v = float(precision_score(y_test, y_pred_v, zero_division=0))
    rec_v = float(recall_score(y_test, y_pred_v))
    f1_v = float(f1_score(y_test, y_pred_v))
    auc_v = float(roc_auc_score(y_test, y_prob_v))

    cm_v = confusion_matrix(y_test, y_pred_v)
    tn_v, fp_v, fn_v, tp_v = map(int, cm_v.ravel())
    spec_v = float(tn_v / (tn_v + fp_v)) if (tn_v + fp_v) > 0 else 0.0

    comp_v = (auc_v * 0.40) + (acc_v * 0.25) + (rec_v * 0.20) + (prec_v * 0.15)

    results['Voting Ensemble'] = {
        'metrics': {
            'accuracy': round(acc_v, 4),
            'precision': round(prec_v, 4),
            'recall': round(rec_v, 4),
            'specificity': round(spec_v, 4),
            'f1_score': round(f1_v, 4),
            'roc_auc': round(auc_v, 4),
            'cv_roc_auc': round((results['Random Forest']['metrics']['cv_roc_auc'] + results['Gradient Boosting']['metrics']['cv_roc_auc'])/2, 4),
            'composite_score': round(comp_v, 4)
        },
        'best_params': {'estimators': ['Random Forest', 'Gradient Boosting', 'XGBoost', 'SVM'], 'voting': 'soft'},
        'confusion_matrix': {'tn': tn_v, 'fp': fp_v, 'fn': fn_v, 'tp': tp_v},
        'feature_importance': results['Random Forest']['feature_importance']
    }

    if comp_v > best_overall_score:
        best_overall_score = comp_v
        best_overall_model = voting_ensemble
        best_overall_name = 'Voting Ensemble'

    # 7. Print Performance Benchmark Table
    print("\n" + "=" * 80)
    print("MODEL BENCHMARK COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Model Name':<22} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8} | {'ROC-AUC':<8} | {'CV-AUC':<8}")
    print("-" * 80)
    for model_name, res in results.items():
        m = res['metrics']
        print(f"{model_name:<22} | {m['accuracy']:.4f}   | {m['precision']:.4f}    | {m['recall']:.4f}   | {m['f1_score']:.4f}   | {m['roc_auc']:.4f}  | {m['cv_roc_auc']:.4f}")
    print("=" * 80)

    # 8. Save Best Model and Metadata
    print(f"\n[BEST PERFORMING MODEL SELECTED]: * {best_overall_name} *")
    print(f"    - Accuracy: {results[best_overall_name]['metrics']['accuracy']*100:.2f}%")
    print(f"    - Precision: {results[best_overall_name]['metrics']['precision']*100:.2f}%")
    print(f"    - Recall: {results[best_overall_name]['metrics']['recall']*100:.2f}%")
    print(f"    - F1-Score: {results[best_overall_name]['metrics']['f1_score']:.4f}")
    print(f"    - ROC-AUC: {results[best_overall_name]['metrics']['roc_auc']:.4f}")

    joblib.dump(best_overall_model, 'saved_models/best_model.joblib')
    summary_data = {
        'best_model_name': best_overall_name,
        'feature_names': feature_cols,
        'results': results
    }
    with open('saved_models/best_model_info.json', 'w') as f:
        json.dump(summary_data, f, indent=2)

    print("\nSaved artifacts:")
    print("  - saved_models/best_model.joblib")
    print("  - saved_models/scaler.joblib")
    print("  - saved_models/best_model_info.json")
    print("=" * 80)
    print("STAGE 2 COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    return results

if __name__ == '__main__':
    train_and_evaluate_models()
