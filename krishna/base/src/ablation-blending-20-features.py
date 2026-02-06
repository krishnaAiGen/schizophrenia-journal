# -*- coding: utf-8 -*-
"""
Ablation Study (20 Features): Is Blending Necessary or Just Complexity?

This script performs model-side ablation with 20 features to answer:
1. Best single model vs Blending (same feature set)
2. Blending without one base learner at a time (leave-one-model-out)
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from scipy.stats import shapiro
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score, roc_curve, auc
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# DATA PREPROCESSING (20 Features Dataset)
# ============================================================
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
# Include all feature columns (original 10 + new 10 handcrafted features)
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
# Only resample columns that exist in the dataframe
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
print(f"Data preprocessing complete. Total subjects: {len(dfs)}")
print(f"Number of features: {sample_df.shape[1]}")

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def calculate_subject_metrics(model, test_dfs, scaler):
    """Calculate metrics at subject level (majority voting on 100 samples per subject)"""
    true_labels = []
    predictions = []
    prediction_probs = []

    for df in test_dfs:
        X_test = df.drop(columns=['class'])
        X_test_scaled = scaler.transform(X_test)

        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

        true_label = df['class'].iloc[0]
        true_labels.append(true_label)

        # Majority voting: if more than 50% predicted as 1, classify as 1
        count_1s = sum(y_pred)
        predictions.append(1 if count_1s > 50 else 0)
        prediction_probs.append(np.mean(y_pred_proba))

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    auc_score = roc_auc_score(true_labels, prediction_probs)

    cm = confusion_matrix(true_labels, predictions)
    TN, FP, FN, TP = cm.ravel()
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    return {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1_score': f1,
        'auc_roc': auc_score,
        'confusion_matrix': cm
    }


def run_blending(train_dfs, test_dfs, base_models_dict, scaler):
    """Run blending ensemble with given base models"""
    # Prepare training data
    full_train_df = pd.concat(train_dfs)
    X_train_all = full_train_df.drop(columns=['class'])
    y_train_all = full_train_df['class'].reset_index(drop=True)
    X_train_scaled = scaler.fit_transform(X_train_all)

    # Generate OOF predictions using Stratified K-Fold
    n_splits = 5
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = {name: np.zeros(len(y_train_all)) for name in base_models_dict}

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train_scaled, y_train_all)):
        X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_tr = y_train_all.iloc[train_idx]

        for name, model in base_models_dict.items():
            model_clone = clone(model)
            model_clone.fit(X_tr, y_tr)
            oof_preds[name][val_idx] = model_clone.predict_proba(X_val)[:, 1]

    # Create meta-features
    meta_train = pd.DataFrame(oof_preds)

    # Train meta-model
    gb_metamodel = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    gb_metamodel.fit(meta_train, y_train_all)

    # Train final base models on full data
    final_models = {}
    for name, model in base_models_dict.items():
        model_clone = clone(model)
        model_clone.fit(X_train_scaled, y_train_all)
        final_models[name] = model_clone

    # Generate test predictions
    meta_test_list = []
    true_labels_test = []

    for df in test_dfs:
        X_test = df.drop(columns=['class'])
        X_test_scaled = scaler.transform(X_test)
        true_label = df['class'].iloc[0]
        true_labels_test.append(true_label)

        test_meta = {}
        for name, model in final_models.items():
            proba = model.predict_proba(X_test_scaled)[:, 1]
            test_meta[name] = np.mean(proba)
        meta_test_list.append(test_meta)

    meta_test = pd.DataFrame(meta_test_list)
    y_true = np.array(true_labels_test)

    # Predict
    y_pred_proba = gb_metamodel.predict_proba(meta_test)[:, 1]
    y_pred_final = gb_metamodel.predict(meta_test)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred_final)
    f1 = f1_score(y_true, y_pred_final)
    auc_score = roc_auc_score(y_true, y_pred_proba)

    cm = confusion_matrix(y_true, y_pred_final)
    TN, FP, FN, TP = cm.ravel()
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    return {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1_score': f1,
        'auc_roc': auc_score,
        'confusion_matrix': cm,
        'y_true': y_true,
        'y_pred_proba': y_pred_proba
    }


# ============================================================
# ABLATION STUDY (20 FEATURES)
# ============================================================
print("\n" + "="*70)
print("ABLATION STUDY (20 FEATURES): IS BLENDING NECESSARY?")
print("="*70)

np.random.seed(42)

# Split data
train_dfs, test_dfs = train_test_split(dfs, test_size=20, random_state=42)
print(f"\nTrain subjects: {len(train_dfs)}, Test subjects: {len(test_dfs)}")

# Prepare training data
full_train_df = pd.concat(train_dfs)
X_train_full = full_train_df.drop(columns=['class'])
y_train_full = full_train_df['class']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full)

# Define all base models with optimized hyperparameters
all_base_models = {
    'RandomForest': RandomForestClassifier(
        n_estimators=152, max_depth=5, min_samples_split=16,
        min_samples_leaf=11, class_weight='balanced', random_state=42, n_jobs=-1
    ),
    'AdaBoost': AdaBoostClassifier(
        n_estimators=320, learning_rate=0.73299, random_state=42
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

# ============================================================
# PART 1: BEST SINGLE MODEL VS BLENDING
# ============================================================
print("\n" + "="*70)
print("PART 1: INDIVIDUAL MODEL PERFORMANCE (20 Features)")
print("="*70)

single_model_results = {}

for name, model in all_base_models.items():
    print(f"\nTraining {name}...")
    model_clone = clone(model)
    model_clone.fit(X_train_scaled, y_train_full)

    metrics = calculate_subject_metrics(model_clone, test_dfs, scaler)
    single_model_results[name] = metrics

    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  F1 Score:    {metrics['f1_score']:.4f}")
    print(f"  AUC-ROC:     {metrics['auc_roc']:.4f}")

# Find best single model
best_single_name = max(single_model_results, key=lambda x: single_model_results[x]['auc_roc'])
best_single_metrics = single_model_results[best_single_name]

print("\n" + "-"*50)
print(f"BEST SINGLE MODEL: {best_single_name}")
print(f"  AUC-ROC: {best_single_metrics['auc_roc']:.4f}")
print("-"*50)

# Run full blending
print("\n" + "="*70)
print("FULL BLENDING ENSEMBLE (All 4 Base Models)")
print("="*70)

blending_models = {
    'rf': all_base_models['RandomForest'],
    'ada': all_base_models['AdaBoost'],
    'gb': all_base_models['GradientBoosting'],
    'xgb': all_base_models['XGBoost']
}

full_blending_results = run_blending(train_dfs, test_dfs, blending_models, StandardScaler())

print(f"\nFull Blending Results:")
print(f"  Accuracy:    {full_blending_results['accuracy']:.4f}")
print(f"  Sensitivity: {full_blending_results['sensitivity']:.4f}")
print(f"  Specificity: {full_blending_results['specificity']:.4f}")
print(f"  F1 Score:    {full_blending_results['f1_score']:.4f}")
print(f"  AUC-ROC:     {full_blending_results['auc_roc']:.4f}")

# ============================================================
# PART 2: LEAVE-ONE-MODEL-OUT ABLATION
# ============================================================
print("\n" + "="*70)
print("PART 2: LEAVE-ONE-MODEL-OUT ABLATION")
print("="*70)
print("(How much does each base model contribute to blending?)")

leave_one_out_results = {}

model_key_mapping = {
    'RandomForest': 'rf',
    'AdaBoost': 'ada',
    'GradientBoosting': 'gb',
    'XGBoost': 'xgb'
}

for removed_model in all_base_models.keys():
    print(f"\nBlending WITHOUT {removed_model}:")

    # Create subset of models excluding the removed one
    subset_models = {
        k: v for k, v in blending_models.items()
        if k != model_key_mapping[removed_model]
    }

    results = run_blending(train_dfs, test_dfs, subset_models, StandardScaler())
    leave_one_out_results[removed_model] = results

    delta_auc = results['auc_roc'] - full_blending_results['auc_roc']

    print(f"  Accuracy:    {results['accuracy']:.4f}")
    print(f"  Sensitivity: {results['sensitivity']:.4f}")
    print(f"  Specificity: {results['specificity']:.4f}")
    print(f"  F1 Score:    {results['f1_score']:.4f}")
    print(f"  AUC-ROC:     {results['auc_roc']:.4f} (Delta: {delta_auc:+.4f})")

# ============================================================
# SUMMARY TABLE
# ============================================================
print("\n" + "="*70)
print("ABLATION STUDY SUMMARY (20 FEATURES)")
print("="*70)

print("\n--- Single Models vs Blending ---")
print(f"{'Model':<20} {'Accuracy':<12} {'Sensitivity':<12} {'Specificity':<12} {'F1 Score':<12} {'AUC-ROC':<12}")
print("-" * 80)

for name, metrics in single_model_results.items():
    print(f"{name:<20} {metrics['accuracy']:<12.4f} {metrics['sensitivity']:<12.4f} {metrics['specificity']:<12.4f} {metrics['f1_score']:<12.4f} {metrics['auc_roc']:<12.4f}")

print(f"{'Full Blending':<20} {full_blending_results['accuracy']:<12.4f} {full_blending_results['sensitivity']:<12.4f} {full_blending_results['specificity']:<12.4f} {full_blending_results['f1_score']:<12.4f} {full_blending_results['auc_roc']:<12.4f}")

print("\n--- Leave-One-Model-Out Impact ---")
print(f"{'Removed Model':<20} {'AUC-ROC':<10} {'Delta AUC':<12} {'F1 Score':<10} {'Delta F1':<12} {'Impact':<20}")
print("-" * 95)

for name, results in leave_one_out_results.items():
    delta_auc = results['auc_roc'] - full_blending_results['auc_roc']
    delta_f1 = results['f1_score'] - full_blending_results['f1_score']
    if delta_auc < -0.02:
        impact = "CRITICAL (hurts)"
    elif delta_auc < 0:
        impact = "Moderate (hurts)"
    elif delta_auc > 0.02:
        impact = "Redundant (helps remove)"
    elif delta_auc > 0:
        impact = "Slightly redundant"
    else:
        impact = "Neutral"

    print(f"{name:<20} {results['auc_roc']:<10.4f} {delta_auc:<+12.4f} {results['f1_score']:<10.4f} {delta_f1:<+12.4f} {impact:<20}")

# ============================================================
# CONCLUSION
# ============================================================
print("\n" + "="*70)
print("CONCLUSION: IS BLENDING NECESSARY? (20 FEATURES)")
print("="*70)

blending_gain = full_blending_results['auc_roc'] - best_single_metrics['auc_roc']

print(f"\nBest Single Model ({best_single_name}): AUC-ROC = {best_single_metrics['auc_roc']:.4f}")
print(f"Full Blending:                        AUC-ROC = {full_blending_results['auc_roc']:.4f}")
print(f"Blending Gain:                        {blending_gain:+.4f}")

if blending_gain > 0.03:
    print("\nVERDICT: Blending provides SIGNIFICANT improvement. Worth the complexity.")
elif blending_gain > 0.01:
    print("\nVERDICT: Blending provides MODEST improvement. Consider trade-off with complexity.")
elif blending_gain > 0:
    print("\nVERDICT: Blending provides MARGINAL improvement. May not justify complexity.")
else:
    print("\nVERDICT: Blending does NOT improve over best single model. Use simpler approach.")

# ============================================================
# VISUALIZATION
# ============================================================
# Create comparison bar chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: AUC-ROC Comparison
models = list(single_model_results.keys()) + ['Full Blending']
auc_scores = [single_model_results[m]['auc_roc'] for m in single_model_results.keys()]
auc_scores.append(full_blending_results['auc_roc'])

colors = ['steelblue'] * len(single_model_results) + ['darkgreen']
bars = axes[0].bar(models, auc_scores, color=colors, edgecolor='black')
axes[0].set_ylabel('AUC-ROC', fontsize=12)
axes[0].set_title('Single Models vs Blending (20 Features)', fontsize=14, fontweight='bold')
axes[0].set_ylim(0.5, 1.0)
axes[0].axhline(y=full_blending_results['auc_roc'], color='darkgreen', linestyle='--', alpha=0.7)
for bar, score in zip(bars, auc_scores):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{score:.3f}', ha='center', va='bottom', fontsize=10)
axes[0].tick_params(axis='x', rotation=45)

# Plot 2: Leave-One-Out Impact
removed_models = list(leave_one_out_results.keys())
deltas = [leave_one_out_results[m]['auc_roc'] - full_blending_results['auc_roc'] for m in removed_models]
colors2 = ['red' if d < 0 else 'green' for d in deltas]

bars2 = axes[1].bar(removed_models, deltas, color=colors2, edgecolor='black')
axes[1].set_ylabel('Change in AUC-ROC', fontsize=12)
axes[1].set_title('Leave-One-Model-Out Impact (20 Features)', fontsize=14, fontweight='bold')
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
for bar, delta in zip(bars2, deltas):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.002 if delta >= 0 else bar.get_height() - 0.015,
                 f'{delta:+.4f}', ha='center', va='bottom' if delta >= 0 else 'top', fontsize=10)
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('/Users/krishnayadav/Documents/test_projects/schizophrenia-journal/krishna/base/result/ablation_blending_comparison_20_features.png', dpi=150)
plt.close()
print("\nSaved: result/ablation_blending_comparison_20_features.png")

# ROC Curves comparison
plt.figure(figsize=(10, 8))

# Plot ROC for best single model
best_model = clone(all_base_models[best_single_name])
best_model.fit(X_train_scaled, y_train_full)

y_true_single = []
y_scores_single = []
for df in test_dfs:
    X_test = df.drop(columns=['class'])
    X_test_scaled = scaler.transform(X_test)
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    y_true_single.append(df['class'].iloc[0])
    y_scores_single.append(np.mean(y_pred_proba))

fpr_single, tpr_single, _ = roc_curve(y_true_single, y_scores_single)
auc_single = auc(fpr_single, tpr_single)

plt.plot(fpr_single, tpr_single, color='blue', lw=2,
         label=f'Best Single ({best_single_name}): AUC = {auc_single:.4f}')

# Plot ROC for blending
fpr_blend, tpr_blend, _ = roc_curve(full_blending_results['y_true'], full_blending_results['y_pred_proba'])
auc_blend = auc(fpr_blend, tpr_blend)

plt.plot(fpr_blend, tpr_blend, color='darkgreen', lw=2,
         label=f'Full Blending: AUC = {auc_blend:.4f}')

plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve: Best Single Model vs Blending (20 Features)', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/Users/krishnayadav/Documents/test_projects/schizophrenia-journal/krishna/base/result/ablation_roc_comparison_20_features.png', dpi=150)
plt.close()
print("Saved: result/ablation_roc_comparison_20_features.png")

print("\n" + "="*70)
print("ABLATION STUDY (20 FEATURES) COMPLETE")
print("="*70)
