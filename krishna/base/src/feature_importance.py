# -*- coding: utf-8 -*-
"""
Feature Importance Analysis (20 Features Dataset)

Uses multiple techniques to calculate feature importance and select top 10 features:
1. Random Forest feature importance
2. XGBoost feature importance
3. Gradient Boosting feature importance
4. Permutation Importance
"""

import numpy as np
import pandas as pd
from scipy.stats import shapiro
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# DATA PREPROCESSING (20 Features Dataset)
# ============================================================
print("\n" + "="*70)
print("LOADING AND PREPROCESSING DATA")
print("="*70)

merged_df = pd.read_csv('/Users/krishnayadav/Documents/test_projects/schizophrenia-journal/data/initial_data_20_features.csv')
df = merged_df.copy()

# Create new features
df['Dynamic range of pupil size'] = (df['PUPIL_SIZE_MAX'] - df['PUPIL_SIZE_MIN']) / df['PUPIL_SIZE_MEAN']
df['Pupil size ratio'] = df['PUPIL_SIZE_MAX'] / df['PUPIL_SIZE_MEAN']

# Drop columns
df = df.drop(columns=['pic_3_3', 'PUPIL_SIZE_MAX', 'PUPIL_SIZE_MIN', 'PUPIL_SIZE_MEAN'])

# Rename columns
new_column_names = {
    'calculated_result': 'Fixation_skewness',
    'CURRENT_FIX_DURATION': 'Valid Viewing Duration',
    'CURRENT_SAC_AVG_VELOCITY': 'Average Saccadic Velocity',
    'CURRENT_SAC_AMPLITUDE': 'Total Saccade Amplitude'
}
df.rename(columns=new_column_names, inplace=True)

# Split into subject dataframes
num_rows = len(df)
chunk_size = 100
num_dataframes = 65
rows_for_dataframes = chunk_size * num_dataframes

dfs = []
for i in range(0, rows_for_dataframes, chunk_size):
    dfs.append(df.iloc[i:i+chunk_size])

df66 = df.iloc[6500:6594]
df67 = df.iloc[6594:6694]
df68 = df.iloc[6694:6794]
df69 = df.iloc[6794:6894]
df70 = df.iloc[6894:6994]
dfs.append(df66)
dfs.append(df67)
dfs.append(df68)
dfs.append(df69)
dfs.append(df70)

# Convert columns to float (including new handcrafted features)
columns_to_convert = ['Average Saccadic Velocity', 'AVERAGE_FIXATION_DURATION', 'AVERAGE_SACCADE_AMPLITUDE',
                      'FSI', 'SER', 'FSI2', 'PRR', 'SFB', 'OEI', 'PIS', 'SVI', 'FPCI', 'AS']
for df in dfs:
    for col in columns_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)

# Handle missing values using Shapiro-Wilk test
columns_to_analyze = ['AVERAGE_FIXATION_DURATION', 'Fixation_skewness', 'Dynamic range of pupil size',
                      'Pupil size ratio', 'Valid Viewing Duration', 'Total Saccade Amplitude',
                      'Average Saccadic Velocity', 'AVERAGE_SACCADE_AMPLITUDE',
                      'FSI', 'SER', 'FSI2', 'PRR', 'SFB', 'OEI', 'PIS', 'SVI', 'FPCI', 'AS']

for i, df in enumerate(dfs):
    for column in columns_to_analyze:
        if column not in df.columns:
            continue
        column_data = df[column]
        column_data_numeric = pd.to_numeric(column_data, errors='coerce')
        if column_data_numeric.dropna().empty:
            continue
        stat, p = shapiro(column_data_numeric.dropna())
        alpha = 0.05
        if p > alpha:
            mean_value = column_data_numeric.mean()
            df[column].fillna(mean_value, inplace=True)
        else:
            median_value = column_data_numeric.median()
            df[column].fillna(median_value, inplace=True)

# Add class labels
class_values = [0] * 30 + [1] * 40
for i, df in enumerate(dfs):
    df['class'] = class_values[i]

# Handle subject 66 (94 rows -> interpolate to 100)
l1 = dfs[65]
new_index = range(6500, 6600)
columns_to_resample = ['Valid Viewing Duration', 'Total Saccade Amplitude',
                       'Average Saccadic Velocity', 'AVERAGE_FIXATION_DURATION',
                       'AVERAGE_SACCADE_AMPLITUDE', 'FIXATION_COUNT',
                       'SACCADE_COUNT', 'Fixation_skewness',
                       'Dynamic range of pupil size', 'Pupil size ratio',
                       'FSI', 'SER', 'FSI2', 'PRR', 'SFB', 'OEI', 'PIS', 'SVI', 'FPCI', 'AS',
                       'class']

l1_resampled = pd.DataFrame(index=new_index)
existing_columns = [col for col in columns_to_resample if col in l1.columns]
l1_resampled[existing_columns] = l1[existing_columns]
l1_resampled = l1_resampled.interpolate(method='linear')
dfs[65] = l1_resampled

dfs[66].index = range(6600, 6700)
dfs[67].index = range(6700, 6800)
dfs[68].index = range(6800, 6900)
dfs[69].index = range(6900, 7000)

dfs[65]['class'] = dfs[65]['class'].astype(int)

# Print feature count
sample_df = dfs[0].drop(columns=['class'])
feature_names = sample_df.columns.tolist()
print(f"\nData preprocessing complete.")
print(f"Total subjects: {len(dfs)}")
print(f"Number of features: {len(feature_names)}")
print(f"Feature names: {feature_names}")

# ============================================================
# PREPARE TRAINING DATA
# ============================================================
print("\n" + "="*70)
print("PREPARING TRAINING DATA")
print("="*70)

np.random.seed(42)

# Split data (use all for training in feature importance)
train_dfs, test_dfs = train_test_split(dfs, test_size=0.2, random_state=42)
print(f"Train subjects: {len(train_dfs)}, Test subjects: {len(test_dfs)}")

# Concatenate all training data
full_train_df = pd.concat(train_dfs)
X_train = full_train_df.drop(columns=['class'])
y_train = full_train_df['class']

full_test_df = pd.concat(test_dfs)
X_test = full_test_df.drop(columns=['class'])
y_test = full_test_df['class']

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# ============================================================
# FEATURE IMPORTANCE CALCULATION
# ============================================================
print("\n" + "="*70)
print("CALCULATING FEATURE IMPORTANCE")
print("="*70)

# Initialize models with optimized hyperparameters
models = {
    'RandomForest': RandomForestClassifier(
        n_estimators=152, max_depth=5, min_samples_split=16,
        min_samples_leaf=11, class_weight='balanced', random_state=42, n_jobs=-1
    ),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
    ),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=42, use_label_encoder=False, eval_metric='logloss'
    )
}

# Dictionary to store feature importances
importance_dict = {}

# 1. Model-based Feature Importances
print("\n--- Training Models for Feature Importance ---")
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    model.fit(X_train_scaled, y_train)

    # Get feature importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        importance_dict[model_name] = importances
        print(f"  ✓ Feature importances extracted")

# 2. Permutation Importance (Model-agnostic)
print("\n--- Calculating Permutation Importance ---")
print("(This may take a few minutes...)")

# Use Random Forest for permutation importance (can use any model)
rf_model = models['RandomForest']
perm_importance = permutation_importance(
    rf_model, X_test_scaled, y_test,
    n_repeats=10, random_state=42, n_jobs=-1
)
importance_dict['Permutation'] = perm_importance.importances_mean

print("  ✓ Permutation importance calculated")

# ============================================================
# AGGREGATE AND RANK FEATURES
# ============================================================
print("\n" + "="*70)
print("FEATURE IMPORTANCE SCORES (ALL 20 FEATURES)")
print("="*70)

# Create DataFrame with all importance scores
importance_df = pd.DataFrame(importance_dict, index=feature_names)

# Calculate average importance across all methods
importance_df['Average'] = importance_df.mean(axis=1)

# Normalize to percentages for easier interpretation
for col in importance_df.columns:
    importance_df[f'{col}_pct'] = (importance_df[col] / importance_df[col].sum()) * 100

# Sort by average importance
importance_df_sorted = importance_df.sort_values('Average', ascending=False)

# Print detailed table
print("\nDetailed Feature Importance Scores:")
print("-" * 120)
print(f"{'Rank':<6} {'Feature':<30} {'RF':<10} {'GB':<10} {'XGB':<10} {'Perm':<10} {'Average':<10}")
print("-" * 120)

for rank, (feature, row) in enumerate(importance_df_sorted.iterrows(), 1):
    print(f"{rank:<6} {feature:<30} {row['RandomForest']:<10.6f} {row['GradientBoosting']:<10.6f} "
          f"{row['XGBoost']:<10.6f} {row['Permutation']:<10.6f} {row['Average']:<10.6f}")

# ============================================================
# TOP 10 FEATURES
# ============================================================
print("\n" + "="*70)
print("TOP 10 MOST IMPORTANT FEATURES")
print("="*70)

top_10_features = importance_df_sorted.head(10)

print("\nTop 10 Features with Percentage Contribution:")
print("-" * 80)
print(f"{'Rank':<6} {'Feature':<30} {'Avg Score':<12} {'Contribution %':<15}")
print("-" * 80)

for rank, (feature, row) in enumerate(top_10_features.iterrows(), 1):
    print(f"{rank:<6} {feature:<30} {row['Average']:<12.6f} {row['Average_pct']:<15.2f}%")

print("\n" + "-" * 80)
cumulative_importance = top_10_features['Average_pct'].sum()
print(f"Cumulative Contribution of Top 10 Features: {cumulative_importance:.2f}%")

# Save top 10 feature names to file
top_10_names = top_10_features.index.tolist()
top_10_df = pd.DataFrame({
    'Rank': range(1, 11),
    'Feature': top_10_names,
    'Importance_Score': top_10_features['Average'].values,
    'Contribution_Percentage': top_10_features['Average_pct'].values
})
top_10_df.to_csv('/Users/krishnayadav/Documents/test_projects/schizophrenia-journal/krishna/base/result/top_10_features.csv', index=False)
print("\nSaved: result/top_10_features.csv")

# ============================================================
# VISUALIZATION
# ============================================================
print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

# 1. Heatmap of Feature Importance across different methods
fig, axes = plt.subplots(2, 1, figsize=(14, 12))

# Prepare data for heatmap (top 15 for better visibility)
top_15 = importance_df_sorted.head(15)
heatmap_data = top_15[['RandomForest', 'GradientBoosting', 'XGBoost', 'Permutation']].T

# Normalize each row for heatmap
heatmap_data_normalized = heatmap_data.div(heatmap_data.sum(axis=1), axis=0) * 100

sns.heatmap(heatmap_data_normalized, annot=True, fmt='.2f', cmap='YlOrRd',
            cbar_kws={'label': 'Importance %'}, ax=axes[0], linewidths=0.5)
axes[0].set_title('Feature Importance Heatmap (Top 15 Features)\nNormalized by Method',
                  fontsize=14, fontweight='bold', pad=20)
axes[0].set_xlabel('Feature', fontsize=11)
axes[0].set_ylabel('Method', fontsize=11)
axes[0].tick_params(axis='x', rotation=45)

# 2. Bar chart of average importance for all features
colors = ['darkgreen' if i < 10 else 'steelblue' for i in range(len(importance_df_sorted))]
bars = axes[1].barh(range(len(importance_df_sorted)), importance_df_sorted['Average'], color=colors, edgecolor='black')
axes[1].set_yticks(range(len(importance_df_sorted)))
axes[1].set_yticklabels(importance_df_sorted.index, fontsize=9)
axes[1].set_xlabel('Average Feature Importance Score', fontsize=11)
axes[1].set_title('Feature Importance Rankings (All 20 Features)\nTop 10 in Green',
                  fontsize=14, fontweight='bold', pad=20)
axes[1].axvline(x=importance_df_sorted['Average'].iloc[9], color='red', linestyle='--',
                linewidth=2, alpha=0.7, label='Top 10 Threshold')
axes[1].legend()
axes[1].invert_yaxis()

# Add values on bars
for i, (bar, val) in enumerate(zip(bars, importance_df_sorted['Average'])):
    axes[1].text(val + 0.001, bar.get_y() + bar.get_height()/2,
                 f'{val:.4f}', va='center', fontsize=8)

plt.tight_layout()
plt.savefig('/Users/krishnayadav/Documents/test_projects/schizophrenia-journal/krishna/base/result/feature_importance_analysis.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: result/feature_importance_analysis.png")

# 3. Comparison bar chart for top 10 across methods
fig, ax = plt.subplots(figsize=(12, 7))

x = np.arange(len(top_10_features))
width = 0.2

methods = ['RandomForest', 'GradientBoosting', 'XGBoost', 'Permutation']
colors_methods = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

for i, (method, color) in enumerate(zip(methods, colors_methods)):
    values = top_10_features[method].values
    ax.bar(x + i * width, values, width, label=method, color=color, edgecolor='black', alpha=0.8)

ax.set_xlabel('Features', fontsize=12, fontweight='bold')
ax.set_ylabel('Importance Score', fontsize=12, fontweight='bold')
ax.set_title('Top 10 Features: Importance Across Different Methods', fontsize=14, fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(top_10_features.index, rotation=45, ha='right', fontsize=10)
ax.legend(loc='upper right', fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/krishnayadav/Documents/test_projects/schizophrenia-journal/krishna/base/result/top_10_features_comparison.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: result/top_10_features_comparison.png")

# ============================================================
# SUMMARY STATISTICS
# ============================================================
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

print(f"\nTotal Features Analyzed: {len(feature_names)}")
print(f"Top 10 Features Selected: {len(top_10_names)}")
print(f"Cumulative Importance of Top 10: {cumulative_importance:.2f}%")

print("\n--- Method Agreement ---")
# Check how many times each feature appears in top 10 across methods
top_10_per_method = {}
for method in methods.keys():
    method_sorted = importance_df.sort_values(method, ascending=False)
    top_10_per_method[method] = set(method_sorted.head(10).index)

# Count appearances
feature_counts = {}
for feature in feature_names:
    count = sum(1 for method_top10 in top_10_per_method.values() if feature in method_top10)
    feature_counts[feature] = count

print("\nTop 10 Features Consensus (appears in X/4 methods' top 10):")
for feature in top_10_names:
    count = feature_counts[feature]
    print(f"  {feature:<35} {count}/4 methods")

print("\n" + "="*70)
print("FEATURE IMPORTANCE ANALYSIS COMPLETE")
print("="*70)
print("\nOutputs:")
print("  1. result/top_10_features.csv")
print("  2. result/feature_importance_analysis.png")
print("  3. result/top_10_features_comparison.png")
