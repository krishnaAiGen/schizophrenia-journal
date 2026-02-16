"""
Greedy Forward Feature Selection with Blending Ensemble
NESTED CV WITH ROBUST IMPROVEMENT CRITERIA (Methodologically Correct)

CRITICAL FIXES:
  ✓ Nested CV: Inner CV on train_dfs only for feature selection
  ✓ Test set never touched until final evaluation
  ✓ Robust improvement criteria: minimum ΔF1 + majority-fold agreement
  ✓ Prevents data leakage and optimistic bias

This script implements greedy forward selection with proper validation:

  OUTER SPLIT (train/test):
    - Training subjects → inner CV for feature selection
    - Test subjects → final holdout evaluation only

  INNER CV (on training subjects only):
    - 5-fold subject-level cross-validation
    - Each candidate evaluated via mean F1 across folds
    - Feature added only if:
        1. Mean ΔF1 ≥ threshold (default 0.01)
        2. Improves in ≥60% of folds (stability)

  FINAL EVALUATION:
    - After feature selection completes
    - Evaluate on test set ONCE
    - This is the unbiased estimate

Experiments:
  - Start_Top1, Start_Top5, Start_Top10, Start_Top15

Author: Krishna Yadav (corrected methodology)
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
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = '/Users/krishnayadav/Documents/test_projects/schizophrenia-journal/data/initial_data_20_features.csv'
RESULT_DIR = '/Users/krishnayadav/Documents/test_projects/schizophrenia-journal/krishna/base/result'

# Nested CV parameters
INNER_CV_FOLDS = 5               # Subject-level K-fold for inner CV
MIN_IMPROVEMENT_THRESHOLD = 0.01 # Minimum mean ΔF1 to accept feature
MIN_FOLD_AGREEMENT = 0.6         # Feature must improve in ≥60% of folds

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
    for i in range(0, 6500, 100):
        dfs.append(df.iloc[i:i + 100].copy())
    dfs.append(df.iloc[6500:6594].copy())
    for start in (6594, 6694, 6794, 6894):
        dfs.append(df.iloc[start:start + 100].copy())
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
    """Train blending ensemble and return all metrics (ONLY for final test evaluation)."""
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
# INNER CV FOR FEATURE SELECTION (KEY FIX)
# ============================================================================

def evaluate_candidate_inner_cv(feature_list, train_dfs, n_folds=5, random_state=42):
    """
    Evaluate a feature set using subject-level K-fold CV on training data ONLY.

    This is the CRITICAL fix: feature selection decisions are based on
    cross-validation within the training set, NOT on the test set.

    Parameters
    ----------
    feature_list : list
        Features to evaluate.
    train_dfs : list of DataFrames
        Training subjects only.
    n_folds : int
        Number of CV folds.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict with:
        'mean_f1': float - mean F1 across folds
        'std_f1': float - std dev of F1 across folds
        'fold_f1s': list - F1 for each fold
        'n_improved': int - number of folds that improved (vs previous)
    """
    # Get subject labels for stratification
    subject_labels = [df['class'].iloc[0] for df in train_dfs]

    # Subject-level K-fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_f1s = []

    for train_idx, val_idx in skf.split(train_dfs, subject_labels):
        # Split subjects into inner train/val
        inner_train = [train_dfs[i] for i in train_idx]
        inner_val   = [train_dfs[i] for i in val_idx]

        # Filter features
        tr = [df[feature_list + ['class']] for df in inner_train]
        val = [df[feature_list + ['class']] for df in inner_val]

        # Train and evaluate
        model = BlendingEnsemble(random_state=random_state)
        model.fit(tr)
        y_true, y_pred, _ = model.get_subject_predictions(val)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        fold_f1s.append(f1)

    return {
        'mean_f1': np.mean(fold_f1s),
        'std_f1': np.std(fold_f1s),
        'fold_f1s': fold_f1s,
    }


# ============================================================================
# FEATURE IMPORTANCE (4 methods + voting)
# ============================================================================

def calculate_voting_importance(X, y, feature_names):
    """Calculate feature importance using 4 methods, rank via voting."""
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
# GREEDY SEARCH WITH NESTED CV (CORRECTED)
# ============================================================================

def run_greedy_experiment_nested_cv(experiment_name, initial_features, feature_pool,
                                     train_dfs, test_dfs, verbose=True):
    """
    Run greedy forward selection with proper nested CV.

    CRITICAL DIFFERENCE FROM ORIGINAL:
      - Feature selection uses INNER CV on train_dfs only
      - test_dfs is NEVER used during selection
      - test_dfs is evaluated ONCE at the end for unbiased estimate

    ROBUST IMPROVEMENT CRITERIA:
      - Accept feature only if:
          1. Mean ΔF1 ≥ MIN_IMPROVEMENT_THRESHOLD
          2. Improves in ≥MIN_FOLD_AGREEMENT of folds

    Returns
    -------
    log : list of dict
        Iteration-by-iteration log with inner CV metrics.
    final_features : list
        Selected features.
    final_test_f1 : float
        Unbiased F1 on test set (computed once at end).
    """
    if verbose:
        print("\n" + "="*70)
        print(f"EXPERIMENT: {experiment_name} (NESTED CV)")
        print("="*70)
        print(f"  Initial features ({len(initial_features)}): {initial_features}")
        print(f"  Pool size: {len(feature_pool)}")
        print(f"  Inner CV: {INNER_CV_FOLDS}-fold | Min ΔF1: {MIN_IMPROVEMENT_THRESHOLD:.4f} | Min fold agreement: {MIN_FOLD_AGREEMENT:.0%}")

    selected = initial_features.copy()
    remaining = feature_pool.copy()

    # Baseline: evaluate initial set via inner CV
    if verbose:
        print("\n  [Baseline] Evaluating initial feature set via inner CV …", end=' ', flush=True)
    baseline_cv = evaluate_candidate_inner_cv(selected, train_dfs, n_folds=INNER_CV_FOLDS)
    current_best_mean_f1 = baseline_cv['mean_f1']
    if verbose:
        print(f"Mean F1 = {current_best_mean_f1:.4f} ± {baseline_cv['std_f1']:.4f}")

    log = [{
        'iteration': 0,
        'action': 'BASELINE',
        'tested_feature': 'N/A',
        'decision': 'N/A',
        'inner_cv_mean_f1': current_best_mean_f1,
        'inner_cv_std_f1': baseline_cv['std_f1'],
        'f1_improvement': 0.0,
        'num_features': len(selected),
        'fold_agreement': 'N/A',
        'selected_features': ', '.join(selected),
    }]

    iteration = 0
    while remaining:
        iteration += 1
        if verbose:
            print(f"\n  [Iteration {iteration}] Testing {len(remaining)} candidates via inner CV …")

        best_candidate = None
        best_mean_f1 = current_best_mean_f1
        best_cv_result = None

        # Evaluate all candidates using INNER CV ONLY (not test set!)
        for candidate in remaining:
            temp_features = selected + [candidate]
            cv_result = evaluate_candidate_inner_cv(temp_features, train_dfs, n_folds=INNER_CV_FOLDS)
            temp_mean_f1 = cv_result['mean_f1']

            if verbose:
                status = "✓" if temp_mean_f1 > best_mean_f1 else "✗"
                print(f"    {candidate:<35} CV F1={temp_mean_f1:.4f}±{cv_result['std_f1']:.4f}  {status}")

            if temp_mean_f1 > best_mean_f1:
                best_candidate = candidate
                best_mean_f1 = temp_mean_f1
                best_cv_result = cv_result

        # Robust improvement check
        if best_candidate is not None:
            improvement = best_mean_f1 - current_best_mean_f1

            # Check 1: Minimum improvement threshold
            passes_threshold = improvement >= MIN_IMPROVEMENT_THRESHOLD

            # Check 2: Fold agreement (how many folds improved?)
            baseline_cv_refold = evaluate_candidate_inner_cv(selected, train_dfs, n_folds=INNER_CV_FOLDS)
            n_improved = sum(1 for new_f1, old_f1 in zip(best_cv_result['fold_f1s'], baseline_cv_refold['fold_f1s']) if new_f1 > old_f1)
            fold_agreement = n_improved / INNER_CV_FOLDS
            passes_agreement = fold_agreement >= MIN_FOLD_AGREEMENT

            if passes_threshold and passes_agreement:
                # ACCEPT
                if verbose:
                    print(f"\n  ✓ KEEP '{best_candidate}'")
                    print(f"    Mean F1: {current_best_mean_f1:.4f} → {best_mean_f1:.4f} (+{improvement:.4f})")
                    print(f"    Fold agreement: {n_improved}/{INNER_CV_FOLDS} ({fold_agreement:.0%}) ✓")

                selected.append(best_candidate)
                remaining.remove(best_candidate)
                current_best_mean_f1 = best_mean_f1

                log.append({
                    'iteration': iteration,
                    'action': 'KEEP',
                    'tested_feature': best_candidate,
                    'decision': 'ACCEPTED',
                    'inner_cv_mean_f1': best_mean_f1,
                    'inner_cv_std_f1': best_cv_result['std_f1'],
                    'f1_improvement': improvement,
                    'num_features': len(selected),
                    'fold_agreement': f"{n_improved}/{INNER_CV_FOLDS}",
                    'selected_features': ', '.join(selected),
                })
            else:
                # REJECT: not robust
                reasons = []
                if not passes_threshold:
                    reasons.append(f"ΔF1={improvement:.4f} < {MIN_IMPROVEMENT_THRESHOLD:.4f}")
                if not passes_agreement:
                    reasons.append(f"fold_agreement={fold_agreement:.0%} < {MIN_FOLD_AGREEMENT:.0%}")

                if verbose:
                    print(f"\n  ✗ REJECT '{best_candidate}' (not robust)")
                    print(f"    Reasons: {'; '.join(reasons)}")
                    print(f"  ✗ STOP — No robust improvement found")
                break
        else:
            if verbose:
                print(f"\n  ✗ STOP — No feature improves mean F1")
            break

    # FINAL TEST EVALUATION (done ONCE, after all selection)
    if verbose:
        print(f"\n  [Final] Evaluating selected {len(selected)} features on HOLDOUT TEST SET …", end=' ', flush=True)
    final_test_metrics = train_and_evaluate(selected, train_dfs, test_dfs)
    final_test_f1 = final_test_metrics['f1_score']
    if verbose:
        print(f"Test F1 = {final_test_f1:.4f} (UNBIASED)")
        print(f"  Features: {selected}")

    # Add final test result to log
    log.append({
        'iteration': 'FINAL',
        'action': 'TEST_EVAL',
        'tested_feature': 'N/A',
        'decision': 'N/A',
        'inner_cv_mean_f1': current_best_mean_f1,
        'inner_cv_std_f1': 'N/A',
        'f1_improvement': 'N/A',
        'num_features': len(selected),
        'fold_agreement': 'N/A',
        'test_f1': final_test_f1,
        'test_accuracy': final_test_metrics['accuracy'],
        'test_precision': final_test_metrics['precision'],
        'test_sensitivity': final_test_metrics['sensitivity'],
        'test_auc_roc': final_test_metrics['auc_roc'],
        'selected_features': ', '.join(selected),
    })

    return log, selected, final_test_f1


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("GREEDY FORWARD SELECTION — NESTED CV (CORRECTED)")
    print("Methodologically sound: inner CV for selection, test for final eval")
    print("="*70)

    # Load data
    print("\n[1] Loading data …")
    df_raw = pd.read_csv(DATA_PATH)
    df_eng = engineer_features(df_raw)

    print("[2] Splitting into 70 subject dataframes …")
    dfs = process_subjects(df_eng)
    dfs = clean_impute_shapiro(dfs)

    # Outer train/test split (72/28)
    np.random.seed(42)
    train_dfs, test_dfs = train_test_split(dfs, train_size=0.72, random_state=42)
    print(f"  Train: {len(train_dfs)} subjects | Test: {len(test_dfs)} subjects (HOLDOUT)")

    # Feature importance
    print("\n[3] Computing voting-based feature importance (on training data) …")
    all_features = [c for c in df_eng.columns if c != 'class']
    full_train = pd.concat(train_dfs)
    X_fi = full_train[all_features].fillna(full_train[all_features].median())
    y_fi = full_train['class']

    imp_df = calculate_voting_importance(X_fi, y_fi, all_features)
    ranked_features = imp_df['Feature'].tolist()

    print(f"\n  Top 10 ranked features:")
    for i, feat in enumerate(ranked_features[:10], 1):
        score = imp_df.loc[imp_df['Feature'] == feat, 'Voting_Score'].values[0]
        print(f"    {i:>2}. {feat:<40} (Voting Score: {score:.2f})")

    imp_df.to_csv(f'{RESULT_DIR}/greedy_corrected_voting_importance.csv', index=False)
    print(f"\n  Saved: {RESULT_DIR}/greedy_corrected_voting_importance.csv")

    # Run experiments
    print("\n" + "="*70)
    print("RUNNING NESTED CV GREEDY EXPERIMENTS")
    print("="*70)

    experiments = []

    # Uncomment to run all experiments
    # for n_init, name in [(1, 'Start_Top1'), (5, 'Start_Top5'), (10, 'Start_Top10'), (15, 'Start_Top15')]:
    #     initial = ranked_features[:n_init]
    #     pool = [f for f in ranked_features if f not in initial]
    #     log, final_feats, final_f1 = run_greedy_experiment_nested_cv(
    #         name, initial, pool, train_dfs, test_dfs, verbose=True)
    #     experiments.append({
    #         'name': name,
    #         'log': log,
    #         'final_features': final_feats,
    #         'final_test_f1': final_f1,
    #         'initial_count': n_init,
    #     })

    # Run only Start_Top15 (as requested)
    exp_initial = ranked_features[:15]
    exp_pool = [f for f in ranked_features if f not in exp_initial]
    log_exp, final_exp, f1_exp = run_greedy_experiment_nested_cv(
        "Start_Top15", exp_initial, exp_pool, train_dfs, test_dfs, verbose=True)
    experiments.append({
        'name': 'Start_Top15',
        'log': log_exp,
        'final_features': final_exp,
        'final_test_f1': f1_exp,
        'initial_count': 15,
    })

    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)

    for exp in experiments:
        log_df = pd.DataFrame(exp['log'])
        log_path = f"{RESULT_DIR}/greedy_corrected_{exp['name']}_log.csv"
        log_df.to_csv(log_path, index=False)
        print(f"  Saved: {log_path}")

        final_feat_df = pd.DataFrame({
            'rank': range(1, len(exp['final_features']) + 1),
            'feature': exp['final_features'],
        })
        feat_path = f"{RESULT_DIR}/greedy_corrected_{exp['name']}_features.csv"
        final_feat_df.to_csv(feat_path, index=False)
        print(f"  Saved: {feat_path}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    summary_rows = []
    for exp in experiments:
        baseline_inner_f1 = exp['log'][0]['inner_cv_mean_f1']
        final_inner_f1 = [r for r in exp['log'] if r['action'] == 'KEEP'][-1]['inner_cv_mean_f1'] if any(r['action'] == 'KEEP' for r in exp['log']) else baseline_inner_f1
        final_test_f1 = exp['final_test_f1']
        n_added = len(exp['final_features']) - exp['initial_count']
        summary_rows.append({
            'Experiment': exp['name'],
            'Initial_Features': exp['initial_count'],
            'Features_Added': n_added,
            'Final_Features': len(exp['final_features']),
            'Baseline_InnerCV_F1': baseline_inner_f1,
            'Final_InnerCV_F1': final_inner_f1,
            'Final_Test_F1_UNBIASED': final_test_f1,
        })

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(f'{RESULT_DIR}/greedy_corrected_summary.csv', index=False)
    print(f"\n  Saved: {RESULT_DIR}/greedy_corrected_summary.csv")

    print("\n" + "="*70)
    print("NESTED CV GREEDY SELECTION COMPLETE")
    print("="*70)
    print("\nKey Points:")
    print("  - Inner CV F1: used for feature selection (training data only)")
    print("  - Test F1: unbiased estimate (computed once at end)")
    print("  - Features selected via robust criteria (ΔF1 threshold + fold agreement)")
    print("  - No data leakage: test set never used during selection")
    print("="*70)


if __name__ == "__main__":
    main()
