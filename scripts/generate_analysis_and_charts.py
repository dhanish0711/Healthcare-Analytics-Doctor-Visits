"""
Healthcare Analytics for Doctor Visits - Comprehensive Analysis & Visualization Pipeline
Incorporating all 34 Project Rubric & Curriculum Requirements
Author: Dhanish Ladwani
Course: VOIS AICTE Batch 1 Data Analytics (DIY Project)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy import stats

def run_comprehensive_visualizations():
    data_path = r"F:\Edunet Data Analytics\diy project\data_cleaned\healthcare_doctor_visits_cleaned.csv"
    raw_path = r"F:\Edunet Data Analytics\diy project\1776250375-P2-Healthcare Analytics for Doctor Visits.csv"
    viz_dir = r"F:\Edunet Data Analytics\diy project\output_visualizations"
    cleaned_dir = r"F:\Edunet Data Analytics\diy project\data_cleaned"
    os.makedirs(viz_dir, exist_ok=True)
    
    print(f"Loading datasets...")
    df_raw = pd.read_csv(raw_path)
    df = pd.read_csv(data_path)
    
    # Configure plotting style
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['axes.labelsize'] = 12
    
    # -------------------------------------------------------------
    # 1. MISSING VALUES HEATMAP
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(df_raw.isnull(), cbar=True, yticklabels=False, cmap='viridis', ax=ax)
    ax.set_title('Missing Values Heatmap (Dataset Audit)', pad=12, fontweight='bold')
    ax.set_xlabel('Dataset Columns')
    ax.set_ylabel('Records (1 to 5,190)')
    plt.tight_layout()
    p1 = os.path.join(viz_dir, "req_missing_values_heatmap.png")
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"Saved: {p1}")
    
    # -------------------------------------------------------------
    # 2. INCOME DISTRIBUTION (MAX, MIN, MEDIAN)
    # -------------------------------------------------------------
    inc_min = df['income'].min()
    inc_max = df['income'].max()
    inc_median = df['income'].median()
    inc_mean = df['income'].mean()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Box plot
    sns.boxplot(x=df['income'], ax=axes[0], color='#38bdf8', notch=True)
    axes[0].set_title('Household Income Box Plot (Min, Median, Max)', fontweight='bold')
    axes[0].set_xlabel('Income ($10k Scale)')
    axes[0].axvline(inc_min, color='red', linestyle='--', label=f'Min: {inc_min:.2f}')
    axes[0].axvline(inc_median, color='darkblue', linestyle='-', linewidth=2, label=f'Median: {inc_median:.2f}')
    axes[0].axvline(inc_max, color='green', linestyle='--', label=f'Max: {inc_max:.2f}')
    axes[0].legend(loc='upper right')
    
    # Histogram with KDE
    sns.histplot(df['income'], kde=True, bins=15, ax=axes[1], color='#0284c7', edgecolor='black')
    axes[1].set_title('Income Distribution with KDE Curve', fontweight='bold')
    axes[1].set_xlabel('Income ($10k Scale)')
    axes[1].set_ylabel('Number of Individuals')
    axes[1].axvline(inc_mean, color='orange', linestyle='--', label=f'Mean: {inc_mean:.2f}')
    axes[1].legend(loc='upper right')
    
    plt.tight_layout()
    p2 = os.path.join(viz_dir, "req_income_min_median_max.png")
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"Saved: {p2}")
    
    # -------------------------------------------------------------
    # 3. COUNT OF PEOPLE BASED ON ILLNESS COUNT
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    ill_counts = df['illness'].value_counts().sort_index()
    bars = ax.bar(ill_counts.index, ill_counts.values, color='#0284c7', edgecolor='black', width=0.6)
    ax.set_title('Total Number of People Based on Illness Count (Past 2 Weeks)', pad=12, fontweight='bold')
    ax.set_xlabel('Number of Illnesses (0 to 5)')
    ax.set_ylabel('Number of People')
    for b in bars:
        h = b.get_height()
        ax.annotate(f'{h:,}\n({h/len(df)*100:.1f}%)', (b.get_x() + b.get_width()/2., h),
                    ha='center', va='bottom', fontsize=10, xytext=(0, 3), textcoords='offset points')
    plt.tight_layout()
    p3 = os.path.join(viz_dir, "req_people_by_illness_count.png")
    plt.savefig(p3, dpi=300)
    plt.close()
    print(f"Saved: {p3}")
    
    # -------------------------------------------------------------
    # 4. REDUCED ACTIVITY DAYS: MALE VS FEMALE BY ILLNESS
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    # Mean reduced days by gender
    gender_red = df.groupby('gender')['reduced'].agg(['mean', 'sum', 'median', 'std']).reset_index()
    sns.barplot(x='gender', y='mean', data=gender_red, ax=axes[0], palette=['#e11d48', '#2563eb'], edgecolor='black')
    axes[0].set_title('Mean Days of Reduced Activity by Gender', fontweight='bold')
    axes[0].set_xlabel('Gender')
    axes[0].set_ylabel('Average Days of Reduced Activity')
    for p in axes[0].patches:
        h = p.get_height()
        axes[0].annotate(f'{h:.3f} days', (p.get_x() + p.get_width()/2., h),
                         ha='center', va='bottom', fontsize=11, fontweight='bold', xytext=(0, 3), textcoords='offset points')
        
    # Grouped by illness and gender
    sns.barplot(x='illness', y='reduced', hue='gender', data=df, ax=axes[1], palette=['#e11d48', '#2563eb'], edgecolor='black')
    axes[1].set_title('Reduced Activity Days by Illness Count & Gender', fontweight='bold')
    axes[1].set_xlabel('Number of Illnesses')
    axes[1].set_ylabel('Mean Reduced Days')
    
    plt.tight_layout()
    p4 = os.path.join(viz_dir, "req_reduced_activity_gender_illness.png")
    plt.savefig(p4, dpi=300)
    plt.close()
    print(f"Saved: {p4}")
    
    # -------------------------------------------------------------
    # 5. INCOME VS HOSPITAL/DOCTOR VISITS ANALYSIS
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    # Regression plot
    sns.regplot(x='income', y='visits', data=df, ax=axes[0],
                scatter_kws={'alpha': 0.25, 'color': '#0284c7'}, line_kws={'color': '#dc2626', 'linewidth': 2.5})
    axes[0].set_title('Income vs. Doctor Visits (Regression Trend)', fontweight='bold')
    axes[0].set_xlabel('Household Income ($10k Scale)')
    axes[0].set_ylabel('Doctor Visits (Past 2 Weeks)')
    
    # Bar plot across income brackets
    sns.barplot(x='income_bracket', y='visits', data=df, ax=axes[1], palette='Blues', edgecolor='black')
    axes[1].set_title('Mean Doctor Consultations by Income Bracket', fontweight='bold')
    axes[1].set_xlabel('Income Bracket')
    axes[1].set_ylabel('Average Visits')
    for p in axes[1].patches:
        h = p.get_height()
        axes[1].annotate(f'{h:.3f}', (p.get_x() + p.get_width()/2., h),
                         ha='center', va='bottom', fontsize=10, xytext=(0, 3), textcoords='offset points')
        
    plt.tight_layout()
    p5 = os.path.join(viz_dir, "req_income_affecting_visits.png")
    plt.savefig(p5, dpi=300)
    plt.close()
    print(f"Saved: {p5}")
    
    # -------------------------------------------------------------
    # 6. COUNT AND VISUALIZE MALES AND FEMALES AFFECTED BY ILLNESS
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    # Countplot of illness distribution by gender
    sns.countplot(x='illness', hue='gender', data=df, ax=axes[0], palette=['#e11d48', '#2563eb'], edgecolor='black')
    axes[0].set_title('People Count Across Illness Levels by Gender', fontweight='bold')
    axes[0].set_xlabel('Number of Illnesses (0 = None, 1-5 = Active)')
    axes[0].set_ylabel('Patient Count')
    
    # Proportion of males and females with active illness (illness > 0)
    df['has_illness'] = (df['illness'] > 0).astype(int)
    ill_pct = df.groupby('gender')['has_illness'].mean() * 100
    bars = axes[1].bar(ill_pct.index, ill_pct.values, color=['#e11d48', '#2563eb'], edgecolor='black', width=0.5)
    axes[1].set_title('% of Males vs Females Experiencing Illness', fontweight='bold')
    axes[1].set_xlabel('Gender')
    axes[1].set_ylabel('% Experiencing at Least 1 Illness')
    axes[1].set_ylim(0, 100)
    for b in bars:
        h = b.get_height()
        axes[1].annotate(f'{h:.1f}%', (b.get_x() + b.get_width()/2., h),
                         ha='center', va='bottom', fontsize=11, fontweight='bold', xytext=(0, 3), textcoords='offset points')
        
    plt.tight_layout()
    p6 = os.path.join(viz_dir, "req_gender_affected_by_illness.png")
    plt.savefig(p6, dpi=300)
    plt.close()
    print(f"Saved: {p6}")
    
    # -------------------------------------------------------------
    # 7. PERCENTAGE OF PEOPLE: GOVT LOW-INCOME, OLD AGE/REPAT, PRIVATE
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    
    pct_freepoor = (df['freepoor'] == 'yes').mean() * 100
    pct_freerepat = (df['freerepat'] == 'yes').mean() * 100
    pct_private = (df['private'] == 'yes').mean() * 100
    pct_uninsured = ((df['freepoor'] == 'no') & (df['freerepat'] == 'no') & (df['private'] == 'no')).mean() * 100
    
    schemes = ['Private Insurance', 'Govt: Veteran/Old Age (freerepat)', 'Govt: Low-Income (freepoor)', 'Standard / Public Only']
    shares = [pct_private, pct_freerepat, pct_freepoor, pct_uninsured]
    colors_pie = ['#3b82f6', '#f59e0b', '#10b981', '#94a3b8']
    
    # Pie Chart
    axes[0].pie(shares, labels=schemes, autopct='%1.2f%%', startangle=140, colors=colors_pie, explode=(0.04, 0.04, 0.08, 0.02), shadow=True)
    axes[0].set_title('Healthcare Insurance & Support Coverage Distribution', fontweight='bold')
    
    # Bar Chart with Counts
    counts = [
        (df['private'] == 'yes').sum(),
        (df['freerepat'] == 'yes').sum(),
        (df['freepoor'] == 'yes').sum(),
        len(df) - ((df['private'] == 'yes') | (df['freerepat'] == 'yes') | (df['freepoor'] == 'yes')).sum()
    ]
    bars_s = axes[1].barh(schemes, shares, color=colors_pie, edgecolor='black')
    axes[1].set_title('Coverage Scheme Percentage & Population Count', fontweight='bold')
    axes[1].set_xlabel('Percentage of Total Population (%)')
    for idx, b in enumerate(bars_s):
        w = b.get_width()
        axes[1].annotate(f'{w:.2f}% ({counts[idx]:,} people)', (w, b.get_y() + b.get_height()/2.),
                         ha='left', va='center', fontsize=10, xytext=(5, 0), textcoords='offset points', fontweight='bold')
    axes[1].set_xlim(0, 55)
    
    plt.tight_layout()
    p7 = os.path.join(viz_dir, "req_insurance_percentages_breakdown.png")
    plt.savefig(p7, dpi=300)
    plt.close()
    print(f"Saved: {p7}")
    
    # -------------------------------------------------------------
    # 8. HORIZONTAL BAR CHART: REDUCED DAYS BY GENDER
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4.5))
    gender_reduced_means = df.groupby('gender')['reduced'].mean()
    gender_colors = ['#e11d48', '#2563eb']
    bars_h = ax.barh(gender_reduced_means.index, gender_reduced_means.values, color=gender_colors, edgecolor='black', height=0.45)
    ax.set_title('Horizontal Bar Chart: Average Days of Reduced Activity by Gender', pad=12, fontweight='bold')
    ax.set_xlabel('Mean Days with Reduced Normal Activity (Past 2 Weeks)')
    ax.set_ylabel('Gender')
    ax.set_xlim(0, 1.5)
    for idx, b in enumerate(bars_h):
        w = b.get_width()
        g_name = gender_reduced_means.index[idx]
        total_days = df[df["gender"] == g_name]["reduced"].sum()
        ax.annotate(f'{w:.3f} days (Total: {total_days:,} days)',
                    (w, b.get_y() + b.get_height()/2.),
                    ha='left', va='center', fontsize=10, fontweight='bold', xytext=(6, 0), textcoords='offset points')
    plt.tight_layout()
    p8 = os.path.join(viz_dir, "req_horizontal_bar_reduced_days_gender.png")
    plt.savefig(p8, dpi=300)
    plt.close()
    print(f"Saved: {p8}")
    
    # -------------------------------------------------------------
    # 9. HORIZONTAL STACKED BAR GRAPH
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5))
    cross_tier = pd.crosstab(df['age_group'], df['visit_frequency_tier'], normalize='index') * 100
    cross_tier = cross_tier[['None (0)', 'Occasional (1-2)', 'Frequent (3+)']]
    cross_tier.plot(kind='barh', stacked=True, color=['#3b82f6', '#10b981', '#ef4444'], edgecolor='black', ax=ax)
    ax.set_title('Horizontal Stacked Bar Chart: Consultation Frequency Tiers Across Age Cohorts', pad=12, fontweight='bold')
    ax.set_xlabel('Percentage of Cohort (%)')
    ax.set_ylabel('Age Cohort')
    ax.legend(title='Visit Frequency Tier', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    p9 = os.path.join(viz_dir, "req_horizontal_stacked_bar_chart.png")
    plt.savefig(p9, dpi=300)
    plt.close()
    print(f"Saved: {p9}")
    
    # -------------------------------------------------------------
    # 10. VERTICAL STACKED BAR CHART WITH MORE THAN 2 DATA SERIES
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    cross_ins = pd.crosstab(df['age_group'], df['insurance_category'], normalize='index') * 100
    colors_ins = ['#0284c7', '#3b82f6', '#f59e0b', '#94a3b8']
    cross_ins.plot(kind='bar', stacked=True, color=colors_ins, edgecolor='black', ax=ax)
    ax.set_title('Vertical Stacked Bar Chart (>2 Data Series): Healthcare Coverage Schemes Across Age Cohorts', pad=12, fontweight='bold')
    ax.set_xlabel('Age Cohort')
    ax.set_ylabel('Proportion of Cohort (%)')
    ax.tick_params(axis='x', rotation=0)
    ax.legend(title='Insurance / Support Scheme', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    p10 = os.path.join(viz_dir, "req_vertical_stacked_bar_more_than_2_data.png")
    plt.savefig(p10, dpi=300)
    plt.close()
    print(f"Saved: {p10}")
    
    # -------------------------------------------------------------
    # 11. 3D PLOTTING (Age vs Income vs Health Impairment)
    # -------------------------------------------------------------
    fig = plt.figure(figsize=(11, 7))
    ax3d = fig.add_subplot(111, projection='3d')
    
    xs = df['age_years']
    ys = df['income']
    zs = df['health_risk_score']
    c_vals = df['visits']
    
    scatter = ax3d.scatter(xs, ys, zs, c=c_vals, cmap='coolwarm', s=25, alpha=0.6, edgecolors='k', linewidth=0.3)
    ax3d.set_title('3D Scatter Plot: Patient Age vs. Income vs. Health Risk Score (Color = Visits)', pad=15, fontweight='bold', fontsize=12)
    ax3d.set_xlabel('Age (Years)', labelpad=10)
    ax3d.set_ylabel('Income ($10k)', labelpad=10)
    ax3d.set_zlabel('Health Risk Impairment (0-100)', labelpad=10)
    ax3d.view_init(elev=20, azim=45)
    
    cbar = plt.colorbar(scatter, ax=ax3d, pad=0.1, shrink=0.7)
    cbar.set_label('Doctor Visits Count', rotation=270, labelpad=15)
    plt.tight_layout()
    p11 = os.path.join(viz_dir, "req_3d_plotting_age_income_health.png")
    plt.savefig(p11, dpi=300)
    plt.close()
    print(f"Saved: {p11}")
    
    # -------------------------------------------------------------
    # 12. MULTI-VARIETY SUBPLOT GALLERY
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Scatter Plot (Age vs Income)
    sns.scatterplot(x='age_years', y='income', hue='gender', data=df, ax=axes[0, 0], palette=['#e11d48', '#2563eb'], alpha=0.5)
    axes[0, 0].set_title('1. Scatter Plot (Age vs Income)', fontweight='bold')
    
    # 2. Bar Graph (Mean Visits by Chronic Status)
    sns.barplot(x='chronic_status', y='visits', hue='chronic_status', data=df, ax=axes[0, 1], palette='Reds', edgecolor='black', legend=False)
    axes[0, 1].set_title('2. Bar Graph (Visits by Chronic Status)', fontweight='bold')
    axes[0, 1].tick_params(axis='x', rotation=15)
    
    # 3. Histogram (Health Score Distribution)
    sns.histplot(df['health'], bins=12, kde=True, ax=axes[0, 2], color='#059669', edgecolor='black')
    axes[0, 2].set_title('3. Histogram with KDE (GHQ Health Score)', fontweight='bold')
    
    # 4. Box Plot (Income across Insurance Categories)
    sns.boxplot(x='insurance_category', y='income', hue='insurance_category', data=df, ax=axes[1, 0], palette='Purples', legend=False)
    axes[1, 0].set_title('4. Box Plot (Income across Insurance)', fontweight='bold')
    axes[1, 0].tick_params(axis='x', rotation=15)
    
    # 5. Line Plot (Reduced Days vs Visits)
    sns.lineplot(x='reduced', y='visits', data=df, ax=axes[1, 1], color='#d97706', marker='o', linewidth=2)
    axes[1, 1].set_title('5. Line Plot (Reduced Days vs Visits)', fontweight='bold')
    
    # 6. Pie Chart (Visit Tiers Proportions)
    tier_counts = df['visit_frequency_tier'].value_counts()
    axes[1, 2].pie(tier_counts, labels=tier_counts.index, autopct='%1.1f%%', colors=['#3b82f6', '#10b981', '#ef4444'], startangle=140)
    axes[1, 2].set_title('6. Pie Chart (Patient Visit Tiers)', fontweight='bold')
    
    plt.suptitle('Multi-Variety Subplot Gallery (6 Diverse Graph Types)', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    p12 = os.path.join(viz_dir, "req_multi_variety_subplots_gallery.png")
    plt.savefig(p12, dpi=300)
    plt.close()
    print(f"Saved: {p12}")
    
    print("\nAll 12 customized visualization requirements generated and saved successfully!")

if __name__ == "__main__":
    run_comprehensive_visualizations()
