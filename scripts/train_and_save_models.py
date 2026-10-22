"""
Healthcare Analytics for Doctor Visits - Machine Learning & Patient Clustering Pipeline
Author: Dhanish Ladwani
Course: VOIS AICTE Batch 1 Data Analytics (DIY Project)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, PoissonRegressor, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
)

def build_ml_pipeline():
    data_path = r"F:\Edunet Data Analytics\diy project\data_cleaned\healthcare_doctor_visits_cleaned.csv"
    models_dir = r"F:\Edunet Data Analytics\diy project\models"
    cleaned_dir = r"F:\Edunet Data Analytics\diy project\data_cleaned"
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"Loading data for ML pipeline: {data_path}")
    df = pd.read_csv(data_path)
    
    # Feature preparation
    feature_cols = [
        'gender_male', 'age_years', 'income',
        'illness', 'reduced', 'health', 'health_risk_score',
        'nchronic_bin', 'lchronic_bin', 'total_chronic_count',
        'private_bin', 'freepoor_bin', 'freerepat_bin'
    ]
    
    df_feat = pd.DataFrame()
    df_feat['gender_male'] = (df['gender'] == 'male').astype(int)
    df_feat['age_years'] = df['age_years']
    df_feat['income'] = df['income']
    df_feat['illness'] = df['illness']
    df_feat['reduced'] = df['reduced']
    df_feat['health'] = df['health']
    df_feat['health_risk_score'] = df['health_risk_score']
    df_feat['nchronic_bin'] = (df['nchronic'] == 'yes').astype(int)
    df_feat['lchronic_bin'] = (df['lchronic'] == 'yes').astype(int)
    df_feat['total_chronic_count'] = df['total_chronic_count']
    df_feat['private_bin'] = (df['private'] == 'yes').astype(int)
    df_feat['freepoor_bin'] = (df['freepoor'] == 'yes').astype(int)
    df_feat['freerepat_bin'] = (df['freerepat'] == 'yes').astype(int)
    
    X = df_feat[feature_cols].copy()
    y_class = df['has_visited'].copy()
    y_reg = df['visits'].copy()
    
    # Stratified Train/Test split
    X_train, X_test, y_train_cls, y_test_cls, y_train_reg, y_test_reg = train_test_split(
        X, y_class, y_reg, test_size=0.20, random_state=42, stratify=y_class
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # -------------------------------------------------------------
    # 1. BINARY CLASSIFICATION (Consultation Likelihood)
    # -------------------------------------------------------------
    print("Training Classification Models...")
    clf_models = {
        'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        'Random Forest Classifier': RandomForestClassifier(n_estimators=150, max_depth=8, class_weight='balanced', random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=120, max_depth=4, random_state=42)
    }
    
    clf_eval = {}
    best_clf_name = None
    best_clf_auc = -1.0
    best_clf = None
    
    for name, model in clf_models.items():
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train_cls)
            preds = model.predict(X_test_scaled)
            probs = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train_cls)
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1]
            
        acc = accuracy_score(y_test_cls, preds)
        prec = precision_score(y_test_cls, preds, zero_division=0)
        rec = recall_score(y_test_cls, preds, zero_division=0)
        f1 = f1_score(y_test_cls, preds, zero_division=0)
        auc = roc_auc_score(y_test_cls, probs)
        cm = confusion_matrix(y_test_cls, preds).tolist()
        
        clf_eval[name] = {
            'accuracy': round(float(acc), 4),
            'precision': round(float(prec), 4),
            'recall': round(float(rec), 4),
            'f1_score': round(float(f1), 4),
            'roc_auc': round(float(auc), 4),
            'confusion_matrix': cm
        }
        print(f"[{name}] Acc: {acc:.4f}, Recall: {rec:.4f}, ROC-AUC: {auc:.4f}")
        
        if auc > best_clf_auc:
            best_clf_auc = auc
            best_clf_name = name
            best_clf = model
            
    # -------------------------------------------------------------
    # 2. COUNT REGRESSION (Consultation Visits)
    # -------------------------------------------------------------
    print("\nTraining Regression Models...")
    reg_models = {
        'Ridge Regression': Ridge(alpha=1.0, random_state=42),
        'Poisson Regressor': PoissonRegressor(alpha=0.1, max_iter=1000),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42)
    }
    
    reg_eval = {}
    best_reg_name = None
    best_reg_mae = 999999.0
    best_reg = None
    
    for name, model in reg_models.items():
        if name in ['Ridge Regression', 'Poisson Regressor']:
            model.fit(X_train_scaled, y_train_reg)
            preds = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train_reg)
            preds = model.predict(X_test)
            
        preds = np.clip(preds, 0, None)
        mae = mean_absolute_error(y_test_reg, preds)
        rmse = np.sqrt(mean_squared_error(y_test_reg, preds))
        r2 = r2_score(y_test_reg, preds)
        
        reg_eval[name] = {
            'mae': round(float(mae), 4),
            'rmse': round(float(rmse), 4),
            'r2_score': round(float(r2), 4)
        }
        print(f"[{name}] MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
        
        if mae < best_reg_mae:
            best_reg_mae = mae
            best_reg_name = name
            best_reg = model
            
    # -------------------------------------------------------------
    # 3. K-MEANS PATIENT CLUSTERING (Unsupervised Risk Segmentation)
    # -------------------------------------------------------------
    print("\nTraining K-Means Clinical Risk Clustering...")
    cluster_features = ['age_years', 'illness', 'reduced', 'health', 'total_chronic_count', 'health_risk_score']
    X_cluster = df_feat[cluster_features]
    scaler_cluster = StandardScaler()
    X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['cluster_id'] = kmeans.fit_predict(X_cluster_scaled)
    
    cluster_summary = {}
    cluster_names = {
        0: 'Routine / Low-Risk Population',
        1: 'Acute Impairment / Transitional',
        2: 'High-Burden Chronic / Vulnerable'
    }
    
    # Sort cluster IDs by average health risk score to align labels meaningfully
    cluster_risk_means = df.groupby('cluster_id')['health_risk_score'].mean().sort_values()
    id_mapping = {old_id: new_id for new_id, old_id in enumerate(cluster_risk_means.index)}
    df['cluster_id_mapped'] = df['cluster_id'].map(id_mapping)
    
    for c_id in range(3):
        subset = df[df['cluster_id_mapped'] == c_id]
        cluster_summary[cluster_names[c_id]] = {
            'patient_count': int(len(subset)),
            'pct_population': round(float(len(subset)/len(df)*100), 1),
            'mean_age': round(float(subset['age_years'].mean()), 1),
            'mean_illness': round(float(subset['illness'].mean()), 2),
            'mean_reduced_days': round(float(subset['reduced'].mean()), 2),
            'mean_health_score': round(float(subset['health'].mean()), 2),
            'mean_risk_score': round(float(subset['health_risk_score'].mean()), 2),
            'mean_doctor_visits': round(float(subset['visits'].mean()), 3),
            'consultation_rate_pct': round(float((subset['visits'] > 0).mean()*100), 1)
        }
        
    print("Cluster Profiles:\n", json.dumps(cluster_summary, indent=2))
    
    # -------------------------------------------------------------
    # 4. FEATURE IMPORTANCES
    # -------------------------------------------------------------
    rf_clf = clf_models['Random Forest Classifier']
    clf_importances = dict(zip(feature_cols, [round(float(v), 4) for v in rf_clf.feature_importances_]))
    clf_importances_sorted = dict(sorted(clf_importances.items(), key=lambda item: item[1], reverse=True))
    
    # -------------------------------------------------------------
    # 5. SERIALIZE MODEL BUNDLE & EVALUATION METRICS
    # -------------------------------------------------------------
    bundle = {
        'feature_cols': feature_cols,
        'scaler': scaler,
        'best_classifier': best_clf,
        'best_classifier_name': best_clf_name,
        'best_regressor': best_reg,
        'best_regressor_name': best_reg_name,
        'kmeans_model': kmeans,
        'scaler_cluster': scaler_cluster,
        'cluster_features': cluster_features,
        'cluster_names': cluster_names,
        'clf_feature_importances': clf_importances_sorted
    }
    
    bundle_path = os.path.join(models_dir, "model_bundle.joblib")
    joblib.dump(bundle, bundle_path)
    print(f"Model bundle saved to: {bundle_path}")
    
    evaluation_summary = {
        'best_classifier': best_clf_name,
        'classification_metrics': clf_eval,
        'best_regressor': best_reg_name,
        'regression_metrics': reg_eval,
        'feature_importances': clf_importances_sorted,
        'cluster_profiles': cluster_summary
    }
    
    eval_path = os.path.join(cleaned_dir, "model_evaluation_metrics.json")
    with open(eval_path, 'w') as f:
        json.dump(evaluation_summary, f, indent=4)
    print(f"Model evaluation saved to: {eval_path}")

if __name__ == "__main__":
    build_ml_pipeline()
