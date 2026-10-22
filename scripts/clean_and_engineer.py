"""
Healthcare Analytics for Doctor Visits - Data Cleaning & Feature Engineering Pipeline
Author: Dhanish Ladwani
Course: VOIS AICTE Batch 1 Data Analytics (DIY Project)
"""

import os
import json
import numpy as np
import pandas as pd

def clean_and_engineer_data():
    raw_path = r"F:\Edunet Data Analytics\diy project\1776250375-P2-Healthcare Analytics for Doctor Visits.csv"
    output_dir = r"F:\Edunet Data Analytics\diy project\data_cleaned"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading raw dataset from: {raw_path}")
    df = pd.read_csv(raw_path)
    
    # 1. Audit and drop redundant ID column
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    initial_shape = df.shape
    print(f"Raw shape: {initial_shape} (Records: {initial_shape[0]}, Features: {initial_shape[1]})")
    
    # Check for missing values
    missing = df.isnull().sum()
    print("Missing values per column:\n", missing[missing > 0])
    
    # 2. Demographic transformations
    # Age is recorded in decimal format (e.g. 0.19 -> 19 years old)
    df['age_years'] = (df['age'] * 100).round().astype(int)
    
    # Age cohorts
    bins_age = [0, 29, 49, 64, 100]
    labels_age = ['Young Adults (18-29)', 'Middle Adults (30-49)', 'Older Adults (50-64)', 'Seniors (65+)']
    df['age_group'] = pd.cut(df['age_years'], bins=bins_age, labels=labels_age, right=True)
    
    # Income brackets (income is scaled, e.g. 0.0 to 1.5)
    bins_income = [-0.01, 0.29, 0.59, 0.99, 2.0]
    labels_income = ['Low (<0.30)', 'Lower-Middle (0.30-0.59)', 'Upper-Middle (0.60-0.99)', 'High (1.00+)']
    df['income_bracket'] = pd.cut(df['income'], bins=bins_income, labels=labels_income)
    
    # 3. Healthcare Support & Insurance Segmentation
    def assign_insurance(row):
        if row['private'] == 'yes':
            return 'Private Insurance'
        elif row['freerepat'] == 'yes':
            return 'Repatriation / Veteran'
        elif row['freepoor'] == 'yes':
            return 'Low-Income Assistance'
        else:
            return 'Public / Standard'
            
    df['insurance_category'] = df.apply(assign_insurance, axis=1)
    
    # 4. Chronic Illness Burden
    def assign_chronic_status(row):
        nc = row['nchronic'] == 'yes'
        lc = row['lchronic'] == 'yes'
        if lc and nc:
            return 'Both Limiting & Non-Limiting'
        elif lc:
            return 'Activity-Limiting Only'
        elif nc:
            return 'Non-Limiting Only'
        else:
            return 'No Chronic Illness'
            
    df['chronic_status'] = df.apply(assign_chronic_status, axis=1)
    df['total_chronic_count'] = ((df['nchronic'] == 'yes').astype(int) + 
                                 (df['lchronic'] == 'yes').astype(int))
    
    # 5. Composite Health Impairment Index (0-100 scale)
    # Weights: Reduced activity days (40%), acute illness count (30%), general health score (20%), chronic penalty (10%)
    reduced_norm = df['reduced'] / 14.0
    illness_norm = df['illness'] / 5.0
    health_norm = df['health'] / 12.0
    chronic_penalty = np.where(df['lchronic'] == 'yes', 1.0, 
                               np.where(df['nchronic'] == 'yes', 0.5, 0.0))
    
    df['health_risk_score'] = (
        (reduced_norm * 40.0) +
        (illness_norm * 30.0) +
        (health_norm * 20.0) +
        (chronic_penalty * 10.0)
    ).round(2)
    
    df['activity_impairment_pct'] = (reduced_norm * 100.0).round(1)
    
    # 6. Target classification & tiering
    df['has_visited'] = (df['visits'] > 0).astype(int)
    
    def assign_visit_tier(v):
        if v == 0:
            return 'None (0)'
        elif v <= 2:
            return 'Occasional (1-2)'
        else:
            return 'Frequent (3+)'
            
    df['visit_frequency_tier'] = df['visits'].apply(assign_visit_tier)
    
    # 7. Reorder columns logically
    ordered_cols = [
        'visits', 'has_visited', 'visit_frequency_tier',
        'gender', 'age', 'age_years', 'age_group',
        'income', 'income_bracket',
        'illness', 'reduced', 'activity_impairment_pct', 'health', 'health_risk_score',
        'nchronic', 'lchronic', 'chronic_status', 'total_chronic_count',
        'private', 'freepoor', 'freerepat', 'insurance_category'
    ]
    df = df[ordered_cols]
    
    # Save cleaned CSV
    output_csv = os.path.join(output_dir, "healthcare_doctor_visits_cleaned.csv")
    df.to_csv(output_csv, index=False)
    print(f"Cleaned dataset saved to: {output_csv} (Shape: {df.shape})")
    
    # Compute summary metrics for downstream tasks
    summary_metrics = {
        "dataset_name": "Healthcare Analytics for Doctor Visits",
        "student_name": "Dhanish Ladwani",
        "total_records": int(len(df)),
        "zero_visits_count": int((df['visits'] == 0).sum()),
        "zero_visits_pct": float(round((df['visits'] == 0).mean() * 100, 2)),
        "one_plus_visits_count": int((df['visits'] > 0).sum()),
        "one_plus_visits_pct": float(round((df['visits'] > 0).mean() * 100, 2)),
        "mean_visits": float(round(df['visits'].mean(), 4)),
        "std_visits": float(round(df['visits'].std(), 4)),
        "max_visits": int(df['visits'].max()),
        "gender_breakdown": {
            "female": {
                "count": int((df['gender'] == 'female').sum()),
                "mean_visits": float(round(df[df['gender'] == 'female']['visits'].mean(), 4)),
                "consultation_rate_pct": float(round((df[df['gender'] == 'female']['visits'] > 0).mean() * 100, 2))
            },
            "male": {
                "count": int((df['gender'] == 'male').sum()),
                "mean_visits": float(round(df[df['gender'] == 'male']['visits'].mean(), 4)),
                "consultation_rate_pct": float(round((df[df['gender'] == 'male']['visits'] > 0).mean() * 100, 2))
            }
        },
        "insurance_breakdown": {
            cat: {
                "count": int((df['insurance_category'] == cat).sum()),
                "mean_visits": float(round(df[df['insurance_category'] == cat]['visits'].mean(), 4)),
                "consultation_rate_pct": float(round((df[df['insurance_category'] == cat]['visits'] > 0).mean() * 100, 2))
            } for cat in df['insurance_category'].unique()
        },
        "chronic_breakdown": {
            status: {
                "count": int((df['chronic_status'] == status).sum()),
                "mean_visits": float(round(df[df['chronic_status'] == status]['visits'].mean(), 4)),
                "consultation_rate_pct": float(round((df[df['chronic_status'] == status]['visits'] > 0).mean() * 100, 2))
            } for status in df['chronic_status'].unique()
        }
    }
    
    summary_json_path = os.path.join(output_dir, "dataset_summary_metrics.json")
    with open(summary_json_path, 'w') as f:
        json.dump(summary_metrics, f, indent=4)
        
    print(f"Summary metrics serialized to: {summary_json_path}")
    return df, summary_metrics

if __name__ == "__main__":
    clean_and_engineer_data()
