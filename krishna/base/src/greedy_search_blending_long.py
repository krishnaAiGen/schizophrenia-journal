"""
Greedy Forward Feature Selection with Blending Ensemble
POOLING EXPERIMENTS

This script runs greedy forward selection experiments starting from different
initial feature set sizes:
  Experiment A: Start with top 1 feature  → greedily add from pool (COMPLETED)
  Experiment B: Start with top 5 features → greedily add from pool (COMPLETED)
  Experiment C: Start with top 10 features → greedily add from pool (COMPLETED)
  Experiment D: Start with top 15 features → greedily add from pool (CURRENT)

For each experiment:
  1. Evaluate baseline with initial feature set
  2. Test ALL remaining features from pool
  3. Add the feature that gives best F1 improvement
  4. Repeat until no feature improves F1
  5. Log every step with detailed metrics

Primary metric: F1 Score
Logs are saved for creating plots later.

Author: Krishna Yadav
Date: 2026-02-14
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from scipy.stats import shapiro
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.base import clone
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# PATHS
# ============================================================================

DATA_PATH = '/Users/krishnayadav/Documents/test_projects/schizophrenia-journal/data/initial_data_20_features.csv'
RESULT_DIR = '/Users/krishnayadav/Documents/test_projects/schizophrenia-journal/krishna/base/result'

# ============================================================================
# DATA PREPROCESSING
# ============================================================================

def engineer_features(df):
    """Feature engineering matching initial_code.py."""
    df_eng = df.copy()
    for col in ['PUPIL_SIZE_MAX', 'PUPIL_SIZE_MIN', 'PUPIL_SIZE_MEAN']:
        df_eng[col] = pd.to_numeric(df_eng[col], errors='coerce')
        df_eng[col] = df_eng[col].fillna(df_eng[col].median())
    df_eng['Dynamic range of pupil size'] = (
        (df_eng['PUPIL_SIZE_MAX'] - df_eng['PUPIL_SIZE_MIN']) / df_eng['PUPIL_SIZE_MEAN'])
    df_eng['Pupil size ratio'] = df_eng['PUPIL_SIZE_MAX'] / df_eng['PUPIL_SIZE_MEAN']
    cols_drop = ['pic_3_3', 'PUPIL_SIZE_MAX', 'PUPIL_SIZE_MIN', 'PUPIL_SIZE_MEAN']
    df_eng = df_eng.drop(columns=[c for c in cols_drop if c in df_eng.columns])
    df_eng = df_eng.rename(columns={
        'calculated_result': 'Fixation_skewness',
        'CURRENT_FIX_DURATION': 'Valid Viewing Duration',
        'CURRENT_SAC_AVG_VELOCITY': 'Average Saccadic Velocity',
        'CURRENT_SAC_AMPLITUDE': 'Total Saccade Amplitude',
    })
    return df_eng


def process_subjects(df):
    """Split into 70 subject dataframes."""
    dfs = []
    # First 65 subjects – 100 rows each
    for i in range(0, 6500, 100):
        dfs.append(df.iloc[i:i + 100].copy())
    # Subject 66: 94 rows
    dfs.append(df.iloc[6500:6594].copy())
    # Subjects 67-70: 100 rows each
    for start in (6594, 6694, 6794, 6894):
        dfs.append(df.iloc[start:start + 100].copy())
    # Class labels
    class_vals = [0] * 30 + [1] * 40
    for i, sdf in enumerate(dfs):
        if i < len(class_vals):
            sdf['class'] = class_vals[i]
    return dfs


def clean_impute_shapiro(dfs):
    """Clean, convert to numeric, impute via Shapiro-Wilk normality test."""
    cols_to_analyze = [
        'AVERAGE_FIXATION_DURATION', 'Fixation_skewness', 'Dynamic range of pupil size',
        'Pupil size ratio', 'Valid Viewing Duration', 'Total Saccade Amplitude',
        'Average Saccadic Velocity', 'AVERAGE_SACCADE_AMPLITUDE',
        'FSI', 'SER', 'FSI2', 'PRR', 'SFB', 'OEI', 'PIS', 'SVI', 'FPCI', 'AS',
        'FIXATION_COUNT', 'SACCADE_COUNT'
    ]
    cleaned = []
    for df in dfs:
        df_c = df.copy()
        for col in cols_to_analyze:
            if col not in df_c.columns:
                continue
            df_c[col] = pd.to_numeric(df_c[col], errors='coerce')
            if df_c[col].isnull().any():
                valid = df_c[col].dropna()
                if len(valid) < 3:
                    df_c[col].fillna(0, inplace=True)
                    continue
                try:
                    stat, p = shapiro(valid)
                    alpha = 0.05
                    if p > alpha:
                        df_c[col].fillna(df_c[col].mean(), inplace=True)
                    else:
                        df_c[col].fillna(df_c[col].median(), inplace=True)
                except Exception:
                    df_c[col].fillna(df_c[col].median(), inplace=True)
        cleaned.append(df_c)
    return cleaned


# ============================================================================
# BLENDING ENSEMBLE
# ============================================================================

class BlendingEnsemble:
    """4-base-model blending ensemble with GBM meta-model."""

    def __init__(self, random_state=42):
        self.rs = random_state
        self.base_models = {
            'rf': RandomForestClassifier(
                n_estimators=152, max_depth=5, min_samples_split=16,
                min_samples_leaf=11, class_weight='balanced',
                random_state=random_state, n_jobs=-1),
            'ada': AdaBoostClassifier(
                n_estimators=320, learning_rate=0.73299, random_state=random_state),
            'gb': GradientBoostingClassifier(
                n_estimators=200, max_depth=5, random_state=random_state),
            'xgb': xgb.XGBClassifier(
                n_estimators=150, max_depth=5, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                random_state=random_state, use_label_encoder=False,
                eval_metric='logloss', n_jobs=-1),
        }
        self.meta_model = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=random_state)
        self.final_base_models = {}
        self.scaler = StandardScaler()

    def fit(self, train_dfs):
        full = pd.concat(train_dfs)
        X = full.drop(columns=['class'])
        y = full['class'].reset_index(drop=True)
        X_sc = self.scaler.fit_transform(X)

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.rs)
        oof = {name: np.zeros(len(y)) for name in self.base_models}
        for tr_idx, va_idx in kf.split(X_sc, y):
            for name, model in self.base_models.items():
                m = clone(model)
                m.fit(X_sc[tr_idx], y.iloc[tr_idx])
                oof[name][va_idx] = m.predict_proba(X_sc[va_idx])[:, 1]

        self.meta_model.fit(pd.DataFrame(oof), y)

        for name, model in self.base_models.items():
            m = clone(model)
            m.fit(X_sc, y)
            self.final_base_models[name] = m

    def get_subject_predictions(self, test_dfs):
        y_true, y_pred, y_prob = [], [], []
        for df in test_dfs:
            X = df.drop(columns=['class'])
            label = int(df['class'].iloc[0])
            X_sc = self.scaler.transform(X)
            meta_row = {name: np.mean(m.predict_proba(X_sc)[:, 1])
                        for name, m in self.final_base_models.items()}
            meta_df = pd.DataFrame([meta_row])
            pred = int(self.meta_model.predict(meta_df)[0])
            prob = float(self.meta_model.predict_proba(meta_df)[:, 1][0])
            y_true.append(label)
            y_pred.append(pred)
            y_prob.append(prob)
        return y_true, y_pred, y_prob


def train_and_evaluate(feature_list, train_dfs, test_dfs):
    """Train blending ensemble and return all metrics."""
    tr = [df[feature_list + ['class']] for df in train_dfs]
    te = [df[feature_list + ['class']] for df in test_dfs]
    model = BlendingEnsemble()
    model.fit(tr)
    y_true, y_pred, y_prob = model.get_subject_predictions(te)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    sens = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.5

    return {
        'accuracy': acc,
        'precision': prec,
        'sensitivity': sens,
        'f1_score': f1,
        'auc_roc': auc,
    }


# ============================================================================
# FEATURE IMPORTANCE (4 methods + voting)
# ============================================================================

def calculate_voting_importance(X, y, feature_names):
    """
    Calculate feature importance using 4 methods, rank via voting.
    Returns DataFrame sorted by average rank (ascending).
    """
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    N = len(feature_names)

    print("  [1/4] Random Forest …", end=' ', flush=True)
    rf = RandomForestClassifier(n_estimators=200, max_depth=None,
                                class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_sc, y)
    imp_rf = rf.feature_importances_
    print("✓")

    print("  [2/4] Gradient Boosting …", end=' ', flush=True)
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                    learning_rate=0.1, random_state=42)
    gb.fit(X_sc, y)
    imp_gb = gb.feature_importances_
    print("✓")

    print("  [3/4] XGBoost …", end=' ', flush=True)
    xgb_m = xgb.XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.1,
                               subsample=0.8, colsample_bytree=0.8,
                               random_state=42, use_label_encoder=False,
                               eval_metric='logloss', n_jobs=-1)
    xgb_m.fit(X_sc, y)
    imp_xgb = xgb_m.feature_importances_
    print("✓")

    print("  [4/4] Permutation Importance …", end=' ', flush=True)
    perm = permutation_importance(rf, X_sc, y, n_repeats=10,
                                  random_state=42, n_jobs=-1)
    imp_perm = np.clip(perm.importances_mean, 0, None)
    print("✓")

    df = pd.DataFrame({
        'Feature':    feature_names,
        'RF_Score':   imp_rf,
        'GB_Score':   imp_gb,
        'XGB_Score':  imp_xgb,
        'Perm_Score': imp_perm,
    })
    for score_col, rank_col in [('RF_Score', 'RF_Rank'), ('GB_Score', 'GB_Rank'),
                                  ('XGB_Score', 'XGB_Rank'), ('Perm_Score', 'Perm_Rank')]:
        df[rank_col] = df[score_col].rank(ascending=False, method='min').astype(int)

    df['Avg_Rank'] = df[['RF_Rank', 'GB_Rank', 'XGB_Rank', 'Perm_Rank']].mean(axis=1)
    df['Voting_Score'] = (N + 1) - df['Avg_Rank']
    df = df.sort_values('Avg_Rank').reset_index(drop=True)
    df.insert(0, 'Rank', range(1, N + 1))
    return df


# ============================================================================
# GREEDY SEARCH EXPERIMENT
# ============================================================================

def run_greedy_experiment(experiment_name, initial_features, feature_pool,
                          train_dfs, test_dfs, verbose=True):
    """
    Run one greedy forward selection experiment.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment (e.g., "Start_Top1").
    initial_features : list
        Starting feature set.
    feature_pool : list
        Pool of candidate features to test.
    train_dfs, test_dfs : list of DataFrames
        Training and test subject data.
    verbose : bool
        Print progress.

    Returns
    -------
    log : list of dict
        Iteration-by-iteration log with F1 scores and decisions.
    final_features : list
        Selected feature set after greedy search.
    final_f1 : float
        Best F1 score achieved.
    """
    if verbose:
        print("\n" + "="*70)
        print(f"EXPERIMENT: {experiment_name}")
        print("="*70)
        print(f"  Initial features ({len(initial_features)}): {initial_features}")
        print(f"  Pool size: {len(feature_pool)}")

    selected = initial_features.copy()
    remaining = feature_pool.copy()

    # Baseline evaluation
    if verbose:
        print("\n  [Baseline] Evaluating initial feature set …", end=' ', flush=True)
    baseline_metrics = train_and_evaluate(selected, train_dfs, test_dfs)
    current_best_f1 = baseline_metrics['f1_score']
    if verbose:
        print(f"F1 = {current_best_f1:.4f}")

    log = [{
        'iteration': 0,
        'action': 'BASELINE',
        'tested_feature': 'N/A',
        'decision': 'N/A',
        'f1_score': current_best_f1,
        'f1_improvement': 0.0,
        'num_features': len(selected),
        'accuracy': baseline_metrics['accuracy'],
        'precision': baseline_metrics['precision'],
        'sensitivity': baseline_metrics['sensitivity'],
        'auc_roc': baseline_metrics['auc_roc'],
        'selected_features': ', '.join(selected),
    }]

    iteration = 0
    while remaining:
        iteration += 1
        if verbose:
            print(f"\n  [Iteration {iteration}] Testing {len(remaining)} candidates from pool …")

        best_candidate = None
        best_f1 = current_best_f1
        best_metrics = None

        # Test all candidates
        for candidate in remaining:
            temp_features = selected + [candidate]
            temp_metrics = train_and_evaluate(temp_features, train_dfs, test_dfs)
            temp_f1 = temp_metrics['f1_score']

            if verbose:
                status = "✓ NEW BEST" if temp_f1 > best_f1 else "✗"
                print(f"    {candidate:<35} F1={temp_f1:.4f}  {status}")

            if temp_f1 > best_f1:
                best_candidate = candidate
                best_f1 = temp_f1
                best_metrics = temp_metrics

        # Decision: keep or stop
        if best_candidate is not None and best_f1 > current_best_f1:
            improvement = best_f1 - current_best_f1
            if verbose:
                print(f"\n  ✓ KEEP '{best_candidate}'  (F1: {current_best_f1:.4f} → {best_f1:.4f}, +{improvement:.4f})")

            selected.append(best_candidate)
            remaining.remove(best_candidate)
            current_best_f1 = best_f1

            log.append({
                'iteration': iteration,
                'action': 'KEEP',
                'tested_feature': best_candidate,
                'decision': 'ACCEPTED',
                'f1_score': best_f1,
                'f1_improvement': improvement,
                'num_features': len(selected),
                'accuracy': best_metrics['accuracy'],
                'precision': best_metrics['precision'],
                'sensitivity': best_metrics['sensitivity'],
                'auc_roc': best_metrics['auc_roc'],
                'selected_features': ', '.join(selected),
            })
        else:
            if verbose:
                print(f"\n  ✗ STOP — No feature improves F1 (current best: {current_best_f1:.4f})")
            break

    if verbose:
        print(f"\n  Final: {len(selected)} features, F1 = {current_best_f1:.4f}")
        print(f"  Features: {selected}")

    return log, selected, current_best_f1


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("GREEDY FORWARD SELECTION — POOLING EXPERIMENTS")
    print("Starting with top 15 features (Top 1, 5, 10 already completed)")
    print("="*70)

    # ---------------------------------------------------------------
    # 1. Load and preprocess data
    # ---------------------------------------------------------------
    print("\n[1] Loading data …")
    df_raw = pd.read_csv(DATA_PATH)
    df_eng = engineer_features(df_raw)

    print("[2] Splitting into 70 subject dataframes …")
    dfs = process_subjects(df_eng)
    dfs = clean_impute_shapiro(dfs)

    # Train/test split (72/28)
    np.random.seed(42)
    train_dfs, test_dfs = train_test_split(dfs, train_size=0.72, random_state=42)
    print(f"  Train: {len(train_dfs)} subjects | Test: {len(test_dfs)} subjects")

    # ---------------------------------------------------------------
    # 2. Feature importance to get voting-ranked features
    # ---------------------------------------------------------------
    print("\n[3] Computing voting-based feature importance (4 methods) …")
    all_features = [c for c in df_eng.columns if c != 'class']
    full_train = pd.concat(train_dfs)
    X_fi = full_train[all_features].fillna(full_train[all_features].median())
    y_fi = full_train['class']

    imp_df = calculate_voting_importance(X_fi, y_fi, all_features)
    ranked_features = imp_df['Feature'].tolist()

    print(f"\n  Top 10 ranked features (by voting):")
    for i, feat in enumerate(ranked_features[:10], 1):
        score = imp_df.loc[imp_df['Feature'] == feat, 'Voting_Score'].values[0]
        print(f"    {i:>2}. {feat:<40} (Voting Score: {score:.2f})")

    # Save feature ranking
    imp_df.to_csv(f'{RESULT_DIR}/greedy_long_voting_importance.csv', index=False)
    print(f"\n  Saved: {RESULT_DIR}/greedy_long_voting_importance.csv")

    # ---------------------------------------------------------------
    # 3. Run greedy experiments
    # ---------------------------------------------------------------
    print("\n" + "="*70)
    print("RUNNING GREEDY EXPERIMENTS")
    print("="*70)

    experiments = []

    # # Experiment A: Start with top 1 feature (COMPLETED - COMMENTED OUT)
    # exp_A_initial = ranked_features[:1]
    # exp_A_pool = [f for f in ranked_features if f not in exp_A_initial]
    # log_A, final_A, f1_A = run_greedy_experiment(
    #     "Start_Top1", exp_A_initial, exp_A_pool, train_dfs, test_dfs, verbose=True)
    # experiments.append({
    #     'name': 'Start_Top1',
    #     'log': log_A,
    #     'final_features': final_A,
    #     'final_f1': f1_A,
    #     'initial_count': 1,
    # })

    # # Experiment B: Start with top 5 features (COMPLETED - COMMENTED OUT)
    # exp_B_initial = ranked_features[:5]
    # exp_B_pool = [f for f in ranked_features if f not in exp_B_initial]
    # log_B, final_B, f1_B = run_greedy_experiment(
    #     "Start_Top5", exp_B_initial, exp_B_pool, train_dfs, test_dfs, verbose=True)
    # experiments.append({
    #     'name': 'Start_Top5',
    #     'log': log_B,
    #     'final_features': final_B,
    #     'final_f1': f1_B,
    #     'initial_count': 5,
    # })

    # # Experiment C: Start with top 10 features (COMPLETED - COMMENTED OUT)
    # exp_C_initial = ranked_features[:10]
    # exp_C_pool = [f for f in ranked_features if f not in exp_C_initial]
    # log_C, final_C, f1_C = run_greedy_experiment(
    #     "Start_Top10", exp_C_initial, exp_C_pool, train_dfs, test_dfs, verbose=True)
    # experiments.append({
    #     'name': 'Start_Top10',
    #     'log': log_C,
    #     'final_features': final_C,
    #     'final_f1': f1_C,
    #     'initial_count': 10,
    # })

    # Experiment D: Start with top 15 features
    exp_D_initial = ranked_features[:15]
    exp_D_pool = [f for f in ranked_features if f not in exp_D_initial]
    log_D, final_D, f1_D = run_greedy_experiment(
        "Start_Top15", exp_D_initial, exp_D_pool, train_dfs, test_dfs, verbose=True)
    experiments.append({
        'name': 'Start_Top15',
        'log': log_D,
        'final_features': final_D,
        'final_f1': f1_D,
        'initial_count': 15,
    })

    # ---------------------------------------------------------------
    # 4. Save individual experiment logs
    # ---------------------------------------------------------------
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)

    for exp in experiments:
        log_df = pd.DataFrame(exp['log'])
        log_path = f"{RESULT_DIR}/greedy_long_{exp['name']}_log.csv"
        log_df.to_csv(log_path, index=False)
        print(f"  Saved: {log_path}")

        # Save final features
        final_feat_df = pd.DataFrame({
            'rank': range(1, len(exp['final_features']) + 1),
            'feature': exp['final_features'],
        })
        feat_path = f"{RESULT_DIR}/greedy_long_{exp['name']}_features.csv"
        final_feat_df.to_csv(feat_path, index=False)
        print(f"  Saved: {feat_path}")

    # ---------------------------------------------------------------
    # 5. Summary comparison table
    # ---------------------------------------------------------------
    print("\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)

    summary_rows = []
    for exp in experiments:
        baseline_f1 = exp['log'][0]['f1_score']
        final_f1 = exp['final_f1']
        improvement = final_f1 - baseline_f1
        n_added = len(exp['final_features']) - exp['initial_count']
        summary_rows.append({
            'Experiment': exp['name'],
            'Initial_Features': exp['initial_count'],
            'Features_Added': n_added,
            'Final_Features': len(exp['final_features']),
            'Baseline_F1': baseline_f1,
            'Final_F1': final_f1,
            'F1_Improvement': improvement,
            'Improvement_Pct': (improvement / baseline_f1) * 100 if baseline_f1 > 0 else 0,
        })

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(f'{RESULT_DIR}/greedy_long_summary.csv', index=False)
    print(f"\n  Saved: {RESULT_DIR}/greedy_long_summary.csv")

    # ---------------------------------------------------------------
    # 6. Visualizations
    # ---------------------------------------------------------------
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)

    # Plot 1: F1 progression for all 3 experiments
    fig, ax = plt.subplots(figsize=(14, 7))
    colors_exp = ['#e74c3c', '#3498db', '#2ecc71']
    markers = ['o', 's', '^']

    for i, exp in enumerate(experiments):
        log_df = pd.DataFrame(exp['log'])
        ax.plot(log_df['num_features'], log_df['f1_score'],
                marker=markers[i], markersize=9, linewidth=2.5,
                color=colors_exp[i], label=exp['name'], alpha=0.85)

    ax.set_xlabel('Number of Features', fontsize=13, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=13, fontweight='bold')
    ax.set_title('Greedy Forward Selection – F1 Score Progression\n3 Pooling Strategies',
                 fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=12, loc='lower right')
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    path1 = f'{RESULT_DIR}/greedy_long_f1_progression.png'
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path1}")

    # Plot 2: Bar chart comparing final F1 scores
    fig, ax = plt.subplots(figsize=(10, 6))
    exp_names = [e['name'] for e in experiments]
    final_f1s = [e['final_f1'] for e in experiments]
    bars = ax.bar(exp_names, final_f1s, color=colors_exp, edgecolor='black',
                  linewidth=1.5, alpha=0.82)

    for bar, f1 in zip(bars, final_f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f'{f1:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Final F1 Score', fontsize=13, fontweight='bold')
    ax.set_title('Greedy Forward Selection – Final F1 Comparison\n3 Pooling Strategies',
                 fontsize=15, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    path2 = f'{RESULT_DIR}/greedy_long_f1_comparison.png'
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path2}")

    # Plot 3: Bar chart showing F1 improvement
    fig, ax = plt.subplots(figsize=(10, 6))
    improvements = [e['final_f1'] - e['log'][0]['f1_score'] for e in experiments]
    bars = ax.bar(exp_names, improvements, color=colors_exp, edgecolor='black',
                  linewidth=1.5, alpha=0.82)

    for bar, imp in zip(bars, improvements):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'+{imp:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('F1 Score Improvement', fontsize=13, fontweight='bold')
    ax.set_title('Greedy Forward Selection – F1 Improvement\n3 Pooling Strategies',
                 fontsize=15, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    path3 = f'{RESULT_DIR}/greedy_long_f1_improvement.png'
    plt.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path3}")

    # ---------------------------------------------------------------
    # 7. Final summary
    # ---------------------------------------------------------------
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*70)
    print("\nOutputs:")
    print("  1. Feature importance: greedy_long_voting_importance.csv")
    print("  2. Experiment logs:")
    for exp in experiments:
        print(f"     - greedy_long_{exp['name']}_log.csv")
        print(f"     - greedy_long_{exp['name']}_features.csv")
    print("  3. Summary: greedy_long_summary.csv")
    print("  4. Plots:")
    print("     - greedy_long_f1_progression.png")
    print("     - greedy_long_f1_comparison.png")
    print("     - greedy_long_f1_improvement.png")

    print("\nBest Result:")
    best_exp = max(experiments, key=lambda e: e['final_f1'])
    print(f"  Experiment: {best_exp['name']}")
    print(f"  Final F1: {best_exp['final_f1']:.4f}")
    print(f"  Final features ({len(best_exp['final_features'])}): {best_exp['final_features']}")
    print("="*70)


if __name__ == "__main__":
    main()
