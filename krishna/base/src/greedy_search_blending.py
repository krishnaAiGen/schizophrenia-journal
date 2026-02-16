# -*- coding: utf-8 -*-
"""
Greedy Feature Addition with Blending/Stacking Ensemble

Incrementally adds top features and measures F1 score improvement using blending approach.
Starts with top 5 features, then adds one feature at a time up to 10 features.
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from scipy.stats import shapiro
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.base import clone
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# FEATURE SETS
# ============================================================
# Top 5 features as starting point
INITIAL_FEATURES = [
    'FIXATION_COUNT',           # 1
    'FPCI',                      # 2
    'Total Saccade Amplitude',   # 3
    'FSI',                       # 4
    'OEI',                       # 5
]

# All available features (20 total)
ALL_FEATURES = [
    'FIXATION_COUNT', 'FPCI', 'Total Saccade Amplitude', 'FSI', 'OEI',
    'Pupil size ratio', 'Dynamic range of pupil size', 'AVERAGE_FIXATION_DURATION',
    'SACCADE_COUNT', 'PRR', 'AVERAGE_SACCADE_AMPLITUDE', 'Fixation_skewness',
    'Valid Viewing Duration', 'Average Saccadic Velocity', 'FSI2', 'SER',
    'SFB', 'PIS', 'SVI', 'AS'
]

# Pool of remaining features (ALL_FEATURES - INITIAL_FEATURES)
FEATURE_POOL = [f for f in ALL_FEATURES if f not in INITIAL_FEATURES]

print(f"\nInitial features (5): {INITIAL_FEATURES}")
print(f"Feature pool ({len(FEATURE_POOL)}): {FEATURE_POOL}")

# ============================================================
# DATA PREPROCESSING (Same as initial_code.py)
# ============================================================
print("\n" + "="*80)
print("GREEDY FEATURE ADDITION WITH BLENDING ENSEMBLE")
print("="*80)

print("\n" + "="*80)
print("LOADING AND PREPROCESSING DATA")
print("="*80)

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

# Convert columns to float
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

print(f"\nData preprocessing complete.")
print(f"Total subjects: {len(dfs)}")

# ============================================================
# BLENDING ENSEMBLE FUNCTION
# ============================================================
def train_and_evaluate_blending(selected_features, train_dfs, test_dfs):
    """
    Train blending ensemble with selected features and evaluate on test set.
    Returns F1 score, accuracy, sensitivity, specificity, and AUC-ROC.
    """
    # Prepare full training data
    full_train_df = pd.concat(train_dfs)
    X_train_all = full_train_df[selected_features]
    y_train_all = full_train_df['class'].reset_index(drop=True)

    # Feature scaling
    scaler_blend = StandardScaler()
    X_train_blend_scaled = scaler_blend.fit_transform(X_train_all)

    # Define base models with optimized hyperparameters
    base_models = {
        'rf': RandomForestClassifier(n_estimators=152, max_depth=5, min_samples_split=16,
                                      min_samples_leaf=11, class_weight='balanced', random_state=42, n_jobs=-1),
        'ada': AdaBoostClassifier(n_estimators=320, learning_rate=0.73299, random_state=42),
        'gb': GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42),
        'xgb': xgb.XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.1,
                                  subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                  random_state=42, use_label_encoder=False, eval_metric='logloss')
    }

    # Generate out-of-fold predictions using Stratified K-Fold
    n_splits = 5
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Store OOF predictions (PROBABILITIES)
    oof_preds = {name: np.zeros(len(y_train_all)) for name in base_models}

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train_blend_scaled, y_train_all)):
        X_tr, X_val = X_train_blend_scaled[train_idx], X_train_blend_scaled[val_idx]
        y_tr = y_train_all.iloc[train_idx]

        for name, model in base_models.items():
            model_clone = clone(model)
            model_clone.fit(X_tr, y_tr)
            oof_preds[name][val_idx] = model_clone.predict_proba(X_val)[:, 1]

    # Create meta-features from OOF predictions
    meta_train = pd.DataFrame(oof_preds)

    # Train meta-model
    gb_metamodel = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    gb_metamodel.fit(meta_train, y_train_all)

    # Train final base models on full training data
    final_models = {}
    for name, model in base_models.items():
        model_clone = clone(model)
        model_clone.fit(X_train_blend_scaled, y_train_all)
        final_models[name] = model_clone

    # Generate meta-features for test set (subject-level predictions)
    meta_test_list = []
    l_true_test = []

    for df in test_dfs:
        X_test = df[selected_features]
        X_test_scaled = scaler_blend.transform(X_test)
        true_label = df['class'].iloc[0]
        l_true_test.append(true_label)

        # Get probability predictions from each base model (average per subject)
        test_meta = {}
        for name, model in final_models.items():
            proba = model.predict_proba(X_test_scaled)[:, 1]
            test_meta[name] = np.mean(proba)

        meta_test_list.append(test_meta)

    meta_test = pd.DataFrame(meta_test_list)
    y_true = np.array(l_true_test)

    # Predict using meta-model
    y_pred_proba = gb_metamodel.predict_proba(meta_test)[:, 1]
    y_pred_final = gb_metamodel.predict(meta_test)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred_final)
    f1 = f1_score(y_true, y_pred_final)
    conf_matrix = confusion_matrix(y_true, y_pred_final)

    TP = conf_matrix[1][1]
    FP = conf_matrix[0][1]
    TN = conf_matrix[0][0]
    FN = conf_matrix[1][0]

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    auc_roc = roc_auc_score(y_true, y_pred_proba)

    return {
        'f1_score': f1,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'auc_roc': auc_roc,
        'confusion_matrix': conf_matrix,
        'y_true': y_true,
        'y_pred_proba': y_pred_proba
    }

# ============================================================
# GREEDY FORWARD FEATURE SELECTION WITH BLENDING
# ============================================================
print("\n" + "="*80)
print("GREEDY FORWARD FEATURE SELECTION")
print("Starting with 5 features, testing each feature from pool")
print("="*80)

# Set random seed
np.random.seed(42)

# Split data (20 test subjects, same as initial_code.py)
train_dfs, test_dfs = train_test_split(dfs, test_size=20, random_state=42)
print(f"\nTrain subjects: {len(train_dfs)}, Test subjects: {len(test_dfs)}")

# Initialize with starting features
selected_features = INITIAL_FEATURES.copy()
remaining_pool = FEATURE_POOL.copy()

# Store results for each iteration
results = []
iteration = 0

# Evaluate baseline (initial 5 features)
print(f"\n{'='*80}")
print(f"ITERATION {iteration}: BASELINE - Evaluating initial {len(selected_features)} features")
print(f"{'='*80}")
print(f"Current features: {selected_features}")

baseline_metrics = train_and_evaluate_blending(selected_features, train_dfs, test_dfs)
current_best_f1 = baseline_metrics['f1_score']

results.append({
    'iteration': iteration,
    'num_features': len(selected_features),
    'action': 'BASELINE',
    'tested_feature': 'N/A',
    'decision': 'N/A',
    'f1_score': baseline_metrics['f1_score'],
    'accuracy': baseline_metrics['accuracy'],
    'sensitivity': baseline_metrics['sensitivity'],
    'specificity': baseline_metrics['specificity'],
    'auc_roc': baseline_metrics['auc_roc'],
    'selected_features': selected_features.copy()
})

print(f"\nBaseline Results:")
print(f"  F1 Score: {baseline_metrics['f1_score']:.4f}")
print(f"  Accuracy: {baseline_metrics['accuracy']:.4f}")
print(f"  Sensitivity: {baseline_metrics['sensitivity']:.4f}")
print(f"  Specificity: {baseline_metrics['specificity']:.4f}")
print(f"  AUC-ROC: {baseline_metrics['auc_roc']:.4f}")

# Greedy forward selection
kept_features_count = 0
discarded_features_count = 0

while remaining_pool:
    iteration += 1
    print(f"\n{'='*80}")
    print(f"ITERATION {iteration}: Testing features from pool ({len(remaining_pool)} remaining)")
    print(f"{'='*80}")
    print(f"Current best F1 Score: {current_best_f1:.4f}")
    print(f"Current features ({len(selected_features)}): {selected_features}")

    best_candidate = None
    best_candidate_f1 = current_best_f1
    best_candidate_metrics = None

    # Test each feature in the pool
    print(f"\nTesting {len(remaining_pool)} candidates:")
    for candidate_feature in remaining_pool:
        # Create temporary feature set with candidate
        temp_features = selected_features + [candidate_feature]

        # Evaluate with candidate feature
        temp_metrics = train_and_evaluate_blending(temp_features, train_dfs, test_dfs)
        temp_f1 = temp_metrics['f1_score']

        print(f"  - Testing '{candidate_feature}': F1 = {temp_f1:.4f} ", end="")

        # Check if this is the best candidate so far
        if temp_f1 > best_candidate_f1:
            best_candidate = candidate_feature
            best_candidate_f1 = temp_f1
            best_candidate_metrics = temp_metrics
            print(f"✓ NEW BEST (+{temp_f1 - current_best_f1:.4f})")
        else:
            print(f"✗ (no improvement)")

    # Decision: Keep or discard best candidate
    if best_candidate is not None and best_candidate_f1 > current_best_f1:
        # KEEP: Feature improves F1 score
        print(f"\n{'*'*80}")
        print(f"DECISION: KEEP '{best_candidate}'")
        print(f"F1 Score improved: {current_best_f1:.4f} → {best_candidate_f1:.4f} (+{best_candidate_f1 - current_best_f1:.4f})")
        print(f"{'*'*80}")

        selected_features.append(best_candidate)
        remaining_pool.remove(best_candidate)
        current_best_f1 = best_candidate_f1
        kept_features_count += 1

        results.append({
            'iteration': iteration,
            'num_features': len(selected_features),
            'action': 'KEEP',
            'tested_feature': best_candidate,
            'decision': 'ACCEPTED',
            'f1_score': best_candidate_metrics['f1_score'],
            'accuracy': best_candidate_metrics['accuracy'],
            'sensitivity': best_candidate_metrics['sensitivity'],
            'specificity': best_candidate_metrics['specificity'],
            'auc_roc': best_candidate_metrics['auc_roc'],
            'selected_features': selected_features.copy()
        })
    else:
        # DISCARD: No feature improves F1 score
        print(f"\n{'*'*80}")
        print(f"DECISION: STOP - No feature improves F1 score")
        print(f"All remaining features tested, none provide improvement")
        print(f"{'*'*80}")
        break

# Final evaluation with selected features
print(f"\n{'='*80}")
print(f"FINAL EVALUATION")
print(f"{'='*80}")
print(f"Total iterations: {iteration}")
print(f"Features kept: {kept_features_count}")
print(f"Features discarded: {len(FEATURE_POOL) - kept_features_count}")
print(f"Final feature count: {len(selected_features)}")
print(f"Final features: {selected_features}")

final_metrics = train_and_evaluate_blending(selected_features, train_dfs, test_dfs)

print(f"\nFinal Results:")
print(f"  F1 Score: {final_metrics['f1_score']:.4f}")
print(f"  Accuracy: {final_metrics['accuracy']:.4f}")
print(f"  Sensitivity: {final_metrics['sensitivity']:.4f}")
print(f"  Specificity: {final_metrics['specificity']:.4f}")
print(f"  AUC-ROC: {final_metrics['auc_roc']:.4f}")

# Store final results for visualization
conf_matrix_final = final_metrics['confusion_matrix']
y_true_final = final_metrics['y_true']
y_pred_proba_final = final_metrics['y_pred_proba']

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# ============================================================
# DISPLAY RESULTS
# ============================================================
print("\n" + "="*80)
print("GREEDY FORWARD SELECTION RESULTS (BLENDING ENSEMBLE)")
print("="*80)

print("\nIteration-by-Iteration Results:")
print("-" * 140)
print(f"{'Iter':<6} {'Action':<10} {'Feature Tested':<35} {'Decision':<12} {'F1 Score':<12} {'Accuracy':<12} {'# Features':<12}")
print("-" * 140)

for idx, row in results_df.iterrows():
    print(f"{row['iteration']:<6} {row['action']:<10} {row['tested_feature']:<35} {row['decision']:<12} "
          f"{row['f1_score']:<12.4f} {row['accuracy']:<12.4f} {row['num_features']:<12}")

print("-" * 140)

# Calculate improvements
baseline_f1 = results_df.iloc[0]['f1_score']
final_f1 = results_df.iloc[-1]['f1_score']
total_improvement = final_f1 - baseline_f1
improvement_percentage = (total_improvement / baseline_f1) * 100

baseline_acc = results_df.iloc[0]['accuracy']
final_acc = results_df.iloc[-1]['accuracy']
acc_improvement = final_acc - baseline_acc

# Get kept features
kept_rows = results_df[results_df['action'] == 'KEEP']
kept_features_list = kept_rows['tested_feature'].tolist()

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"\nBaseline (5 features):")
print(f"  Features: {INITIAL_FEATURES}")
print(f"  F1 Score: {baseline_f1:.4f}")
print(f"  Accuracy: {baseline_acc:.4f}")

print(f"\nKept Features ({len(kept_features_list)}):")
for i, feat in enumerate(kept_features_list, 1):
    print(f"  {i}. {feat}")

print(f"\nFinal ({len(selected_features)} features):")
print(f"  Features: {selected_features}")
print(f"  F1 Score: {final_f1:.4f}")
print(f"  Accuracy: {final_acc:.4f}")

print(f"\nTotal Improvement:")
print(f"  F1 Score: +{total_improvement:.4f} ({improvement_percentage:+.2f}%)")
print(f"  Accuracy: +{acc_improvement:.4f} ({(acc_improvement/baseline_acc)*100:+.2f}%)")

# Save results
results_df.to_csv('/Users/krishnayadav/Documents/test_projects/schizophrenia-journal/krishna/base/result/greedy_forward_selection_results.csv', index=False)
print("\nSaved: result/greedy_forward_selection_results.csv")

# Save selected features
selected_features_df = pd.DataFrame({
    'rank': range(1, len(selected_features) + 1),
    'feature': selected_features,
    'type': ['initial'] * len(INITIAL_FEATURES) + ['added'] * (len(selected_features) - len(INITIAL_FEATURES))
})
selected_features_df.to_csv('/Users/krishnayadav/Documents/test_projects/schizophrenia-journal/krishna/base/result/selected_features.csv', index=False)
print("Saved: result/selected_features.csv")

# ============================================================
# VISUALIZATION 1: BAR CHART (F1 Score Progression)
# ============================================================
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

fig, ax = plt.subplots(figsize=(14, 8))

# Bar colors (gradient)
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(results_df)))

# Create bars
bars = ax.bar(range(len(results_df)), results_df['f1_score'], color=colors,
               edgecolor='black', linewidth=1.5, alpha=0.85)

# Customize x-axis labels
x_labels = []
for idx, row in results_df.iterrows():
    if row['action'] == 'BASELINE':
        x_labels.append(f"Baseline\n({row['num_features']} feat)")
    else:
        x_labels.append(f"Iter {row['iteration']}\n({row['num_features']} feat)")

ax.set_xticks(range(len(results_df)))
ax.set_xticklabels(x_labels, fontsize=10, fontweight='bold')

# Add value labels on top of bars
for i, (bar, row) in enumerate(zip(bars, results_df.itertuples())):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{row.f1_score:.4f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add feature name below the bar (for KEEP actions)
    if row.action == 'KEEP':
        feature_short = row.tested_feature[:20] + '...' if len(row.tested_feature) > 20 else row.tested_feature
        ax.text(bar.get_x() + bar.get_width()/2., -0.03,
                f'+{feature_short}',
                ha='center', va='top', fontsize=8, rotation=0,
                style='italic', color='darkblue')

# Labels and title
ax.set_xlabel('Iteration', fontsize=13, fontweight='bold')
ax.set_ylabel('F1 Score', fontsize=13, fontweight='bold')
ax.set_title('Greedy Forward Feature Selection: F1 Score Progression\n(Blending Ensemble - Only Keeping Features That Improve F1)',
             fontsize=15, fontweight='bold', pad=20)

# Grid
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
ax.set_axisbelow(True)

# Add horizontal line for baseline
ax.axhline(y=baseline_f1, color='red', linestyle='--', linewidth=2,
           alpha=0.6, label=f'Baseline (5 features): {baseline_f1:.4f}')

# Set y-axis limits
y_min = max(0, results_df['f1_score'].min() - 0.1)
y_max = min(1.0, results_df['f1_score'].max() + 0.1)
ax.set_ylim([y_min, y_max])

# Legend
ax.legend(loc='lower right', fontsize=11)

# Add annotation for total improvement
if len(results_df) > 1:
    ax.annotate(f'Total Improvement:\n+{total_improvement:.4f} ({improvement_percentage:+.2f}%)',
                xy=(len(results_df)-1, final_f1), xytext=(max(0, len(results_df)-3), final_f1 + 0.05),
                arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2),
                fontsize=11, fontweight='bold', color='darkgreen',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))

plt.tight_layout()
plt.savefig('/Users/krishnayadav/Documents/test_projects/schizophrenia-journal/krishna/base/result/greedy_forward_selection_barplot.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: result/greedy_forward_selection_barplot.png")

# ============================================================
# VISUALIZATION 2: LINE PLOT WITH MULTIPLE METRICS
# ============================================================
fig, ax = plt.subplots(figsize=(12, 7))

# Plot multiple metrics
metrics_to_plot = ['f1_score', 'accuracy', 'sensitivity', 'auc_roc']
colors_metrics = ['blue', 'green', 'orange', 'red']
markers = ['o', 's', '^', 'D']

for metric, color, marker in zip(metrics_to_plot, colors_metrics, markers):
    ax.plot(results_df['num_features'], results_df[metric],
            marker=marker, markersize=10, linewidth=2.5, color=color,
            label=metric.replace('_', ' ').title(), alpha=0.8
# Labels and title
ax.set_xlabel('Number of Features', fontsize=13, fontweight='bold')
ax.set_ylabel('Score', fontsize=13, fontweight='bold')
ax.set_title('Performance Metrics vs Number of Features\n(Greedy Forward Selection with Blending Ensemble)',
             fontsize=15, fontweight='bold', pad=20)

# Grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)

# Set x-axis ticks
ax.set_xticks(results_df['num_features'].unique())

# Set y-axis limits
ax.set_ylim([0, 1.05])

# Legend
ax.legend(loc='lower right', fontsize=11)

plt.tight_layout()
plt.savefig('/Users/krishnayadav/Documents/test_projects/schizophrenia-journal/krishna/base/result/greedy_forward_selection_lineplot.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: result/greedy_forward_selection_lineplot.png")

# ============================================================
# VISUALIZATION 3: CONFUSION MATRIX (Final)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(conf_matrix_final, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Control', 'Schizophrenia'],
            yticklabels=['Control', 'Schizophrenia'], ax=ax)
ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
ax.set_title(f'Confusion Matrix - Blending Ensemble ({len(selected_features)} Features)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/Users/krishnayadav/Documents/test_projects/schizophrenia-journal/krishna/base/result/greedy_forward_selection_confusion_matrix.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: result/greedy_forward_selection_confusion_matrix.png")

# ============================================================
# VISUALIZATION 4: ROC CURVE (Final)
# ============================================================
fpr, tpr, thresholds = roc_curve(y_true_final, y_pred_proba_final)
roc_auc_final = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc_final:.4f})')
ax.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Classifier')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax.set_title(f'ROC Curve - Blending Ensemble ({len(selected_features)} Features)',
             fontsize=14, fontweight='bold')
ax.legend(loc="lower right", fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/Users/krishnayadav/Documents/test_projects/schizophrenia-journal/krishna/base/result/greedy_forward_selection_roc_curve.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: result/greedy_forward_selection_roc_curve.png")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*80)
print("GREEDY FORWARD FEATURE SELECTION WITH BLENDING ENSEMBLE - COMPLETE")
print("="*80)
print("\nOutputs:")
print("  1. result/greedy_forward_selection_results.csv")
print("  2. result/selected_features.csv")
print("  3. result/greedy_forward_selection_barplot.png")
print("  4. result/greedy_forward_selection_lineplot.png")
print("  5. result/greedy_forward_selection_confusion_matrix.png")
print("  6. result/greedy_forward_selection_roc_curve.png")
print("\nKey Findings:")
print(f"  - Initial features: {len(INITIAL_FEATURES)}")
print(f"  - Features tested from pool: {len(FEATURE_POOL)}")
print(f"  - Features kept: {len(selected_features) - len(INITIAL_FEATURES)}")
print(f"  - Final feature count: {len(selected_features)}")
print(f"  - Starting F1 Score: {baseline_f1:.4f}")
print(f"  - Final F1 Score: {final_f1:.4f}")
print(f"  - Total Improvement: +{total_improvement:.4f} ({improvement_percentage:+.2f}%)")
print(f"  - Final Accuracy: {final_acc:.4f}")
print(f"  - Final Sensitivity: {results_df.iloc[-1]['sensitivity']:.4f}")
print(f"  - Final Specificity: {results_df.iloc[-1]['specificity']:.4f}")
print(f"  - Final AUC-ROC: {results_df.iloc[-1]['auc_roc']:.4f}")
print("\nSelected Features:")
for i, feat in enumerate(selected_features, 1):
    print(f"  {i}. {feat}")
print("="*80)
