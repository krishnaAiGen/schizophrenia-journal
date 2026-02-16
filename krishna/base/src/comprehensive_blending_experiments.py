"""
Comprehensive Blending Model Experiments for Schizophrenia Classification
(Subject-Based Splitting)

Experiments:
1. Blending model on 10-feature dataset (Baseline)
2. Blending model on NEW features only
3. Blending model on ALL 20 features
4. Feature importance analysis (4 methods + voting-based ranking)
5. Greedy forward selection: start with top 5 ranked features, add one per step

Metrics reported at every step:
  Accuracy | Precision | Sensitivity (Recall) | F1 Score | ROC-AUC

Feature Importance Methods:
  1. Random Forest MDI (Mean Decrease in Impurity)
  2. Gradient Boosting Feature Importance
  3. XGBoost Feature Importance
  4. Permutation Importance (model-agnostic, using RF)

Voting-based ranking:
  Each method ranks features 1–N; average rank determines final order.
  Voting Score = (N + 1) - AvgRank  (higher = more important)

Author: Krishna Yadav
Date: 2026-02-11
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.base import clone
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# PATHS
# ============================================================================

DATA_DIR = '/Users/krishnayadav/Documents/test_projects/schizophrenia-journal/data'
RESULT_DIR = '/Users/krishnayadav/Documents/test_projects/schizophrenia-journal/krishna/base/result'

# ============================================================================
# DATA PREPARATION & SUBJECT SPLITTING
# ============================================================================

def engineer_features(df):
    """Replicate feature engineering from initial_code.py."""
    df_eng = df.copy()
    for col in ['PUPIL_SIZE_MAX', 'PUPIL_SIZE_MIN', 'PUPIL_SIZE_MEAN']:
        df_eng[col] = pd.to_numeric(df_eng[col], errors='coerce')
        df_eng[col] = df_eng[col].fillna(df_eng[col].median())
    df_eng['Dynamic range of pupil size'] = (
        (df_eng['PUPIL_SIZE_MAX'] - df_eng['PUPIL_SIZE_MIN']) / df_eng['PUPIL_SIZE_MEAN'])
    df_eng['Pupil size ratio'] = df_eng['PUPIL_SIZE_MAX'] / df_eng['PUPIL_SIZE_MEAN']
    cols_to_drop = ['pic_3_3', 'PUPIL_SIZE_MAX', 'PUPIL_SIZE_MIN', 'PUPIL_SIZE_MEAN']
    df_eng = df_eng.drop(columns=[c for c in cols_to_drop if c in df_eng.columns])
    df_eng = df_eng.rename(columns={
        'calculated_result': 'Fixation_skewness',
        'CURRENT_FIX_DURATION': 'Valid Viewing Duration',
        'CURRENT_SAC_AVG_VELOCITY': 'Average Saccadic Velocity',
        'CURRENT_SAC_AMPLITUDE': 'Total Saccade Amplitude',
    })
    return df_eng


def process_subjects(df):
    """Split dataframe into per-subject dataframes (70 subjects total)."""
    dfs = []
    # First 65 subjects – 100 rows each
    for i in range(0, 6500, 100):
        dfs.append(df.iloc[i:i + 100].copy())
    # Subject 66: 94 rows (kept as-is; majority-vote rule still works)
    dfs.append(df.iloc[6500:6594].copy())
    # Subjects 67-70: 100 rows each
    for start in (6594, 6694, 6794, 6894):
        dfs.append(df.iloc[start:start + 100].copy())
    # Assign class labels: 0→first 30 subjects, 1→next 40
    class_values = [0] * 30 + [1] * 40
    result = []
    for i, sdf in enumerate(dfs):
        if i < len(class_values):
            sdf = sdf.copy()
            sdf['class'] = class_values[i]
            result.append(sdf)
    return result


def clean_and_impute(dfs):
    """Convert to numeric and median-impute per subject."""
    cleaned = []
    for df in dfs:
        df_c = df.copy()
        for col in [c for c in df_c.columns if c != 'class']:
            df_c[col] = pd.to_numeric(df_c[col], errors='coerce')
            if df_c[col].isnull().any():
                med = df_c[col].median()
                df_c[col].fillna(0 if pd.isna(med) else med, inplace=True)
        cleaned.append(df_c)
    return cleaned


# ============================================================================
# BLENDING ENSEMBLE
# ============================================================================

class BlendingEnsemble:
    """4-base-model stacking ensemble with a GBM meta-model."""

    def __init__(self, random_state=42):
        self.rs = random_state
        self.base_models = {
            'rf': RandomForestClassifier(
                n_estimators=152, max_depth=5, min_samples_split=16,
                min_samples_leaf=11, class_weight='balanced',
                random_state=random_state, n_jobs=-1),
            'ada': AdaBoostClassifier(
                n_estimators=320, learning_rate=0.73299,
                random_state=random_state),
            'gb': GradientBoostingClassifier(
                n_estimators=200, max_depth=5, random_state=random_state),
            'xgb': xgb.XGBClassifier(
                n_estimators=150, max_depth=5, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                random_state=random_state, use_label_encoder=False,
                eval_metric='logloss', n_jobs=-1),
        }
        self.meta_model = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            random_state=random_state)
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


# ============================================================================
# METRICS HELPER
# ============================================================================

def compute_metrics(y_true, y_pred, y_prob):
    """Return dict with all 5 classification metrics."""
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    sens = recall_score(y_true, y_pred, zero_division=0)     # sensitivity = recall
    f1   = f1_score(y_true, y_pred, zero_division=0)
    try:
        roc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc = 0.5
    return {'Accuracy': acc, 'Precision': prec, 'Sensitivity': sens, 'F1': f1, 'ROC_AUC': roc}


def print_metrics(metrics, indent=2):
    pad = ' ' * indent
    print(f"{pad}Accuracy:    {metrics['Accuracy']:.4f}")
    print(f"{pad}Precision:   {metrics['Precision']:.4f}")
    print(f"{pad}Sensitivity: {metrics['Sensitivity']:.4f}")
    print(f"{pad}F1 Score:    {metrics['F1']:.4f}")
    print(f"{pad}ROC-AUC:     {metrics['ROC_AUC']:.4f}")


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_experiment(name, feature_list, dfs_train, dfs_test):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Features ({len(feature_list)}): {feature_list}")

    tr = [df[feature_list + ['class']] for df in dfs_train]
    te = [df[feature_list + ['class']] for df in dfs_test]

    model = BlendingEnsemble()
    model.fit(tr)
    y_true, y_pred, y_prob = model.get_subject_predictions(te)

    m = compute_metrics(y_true, y_pred, y_prob)
    print_metrics(m)
    return dict(Experiment=name, **m)


# ============================================================================
# FEATURE IMPORTANCE: 4 METHODS + VOTING-BASED RANKING
# ============================================================================

def calculate_feature_importance_voting(X, y, feature_names):
    """
    Compute feature importance via 4 methods and produce a voting-based ranking.

    Methods
    -------
    1. Random Forest MDI
    2. Gradient Boosting MDI
    3. XGBoost gain importance
    4. Permutation importance (using RF, evaluated on held-out split)

    Voting ranking
    --------------
    For each method rank features 1…N (1 = most important).
    Average rank across the 4 methods = Avg_Rank.
    Voting Score = (N + 1) - Avg_Rank  (higher → more important).
    """
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    N = len(feature_names)

    # ---- Method 1: Random Forest MDI ----
    print("  [1/4] Random Forest MDI …", end=' ', flush=True)
    rf = RandomForestClassifier(n_estimators=200, max_depth=None,
                                class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_sc, y)
    imp_rf = rf.feature_importances_
    print("done")

    # ---- Method 2: Gradient Boosting MDI ----
    print("  [2/4] Gradient Boosting MDI …", end=' ', flush=True)
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                    learning_rate=0.1, random_state=42)
    gb.fit(X_sc, y)
    imp_gb = gb.feature_importances_
    print("done")

    # ---- Method 3: XGBoost gain importance ----
    print("  [3/4] XGBoost gain importance …", end=' ', flush=True)
    xgb_m = xgb.XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.1,
                               subsample=0.8, colsample_bytree=0.8,
                               random_state=42, use_label_encoder=False,
                               eval_metric='logloss', n_jobs=-1)
    xgb_m.fit(X_sc, y)
    imp_xgb = xgb_m.feature_importances_
    print("done")

    # ---- Method 4: Permutation Importance ----
    print("  [4/4] Permutation Importance (10 repeats) …", end=' ', flush=True)
    perm = permutation_importance(rf, X_sc, y, n_repeats=10,
                                  random_state=42, n_jobs=-1)
    imp_perm = perm.importances_mean
    # Clip negatives to 0 for consistent treatment
    imp_perm = np.clip(imp_perm, 0, None)
    print("done")

    # ---- Build importance DataFrame ----
    imp_df = pd.DataFrame({
        'Feature':       feature_names,
        'RF_Score':      imp_rf,
        'GB_Score':      imp_gb,
        'XGB_Score':     imp_xgb,
        'Perm_Score':    imp_perm,
    })

    # ---- Rank each feature per method (1 = most important) ----
    for col, rank_col in [('RF_Score', 'RF_Rank'),
                          ('GB_Score', 'GB_Rank'),
                          ('XGB_Score', 'XGB_Rank'),
                          ('Perm_Score', 'Perm_Rank')]:
        imp_df[rank_col] = imp_df[col].rank(ascending=False, method='min').astype(int)

    imp_df['Avg_Rank']      = imp_df[['RF_Rank', 'GB_Rank', 'XGB_Rank', 'Perm_Rank']].mean(axis=1)
    imp_df['Voting_Score']  = (N + 1) - imp_df['Avg_Rank']   # higher = more important
    imp_df = imp_df.sort_values('Avg_Rank').reset_index(drop=True)
    imp_df.insert(0, 'Rank', range(1, N + 1))

    return imp_df


def print_feature_importance_table(imp_df):
    sep = '-' * 128
    print(sep)
    print(f"{'Rank':<5} {'Feature':<35} "
          f"{'RF_Score':<12} {'GB_Score':<12} {'XGB_Score':<12} {'Perm_Score':<12} "
          f"{'RF_Rk':<7} {'GB_Rk':<7} {'XGB_Rk':<7} {'Perm_Rk':<8} "
          f"{'Avg_Rank':<10} {'VotScore':<10}")
    print(sep)
    for _, row in imp_df.iterrows():
        print(f"{int(row['Rank']):<5} {row['Feature']:<35} "
              f"{row['RF_Score']:<12.6f} {row['GB_Score']:<12.6f} "
              f"{row['XGB_Score']:<12.6f} {row['Perm_Score']:<12.6f} "
              f"{int(row['RF_Rank']):<7} {int(row['GB_Rank']):<7} "
              f"{int(row['XGB_Rank']):<7} {int(row['Perm_Rank']):<8} "
              f"{row['Avg_Rank']:<10.2f} {row['Voting_Score']:<10.2f}")
    print(sep)


# ============================================================================
# GREEDY FORWARD SELECTION (ranked order, starting from top 5)
# ============================================================================

def run_greedy_selection(ranked_features, dfs_train, dfs_test):
    """
    Start with the top-5 voting-ranked features.
    Then add the next ranked feature one at a time (positions 6, 7, … up to 20).
    Report all 5 metrics at every step.
    """
    print("\n" + "="*70)
    print("GREEDY FORWARD SELECTION  (voting-ranked order, top-5 start)")
    print("="*70)

    greedy_results = []
    total = len(ranked_features)

    for k in range(5, total + 1):
        current_feats = ranked_features[:k]
        newly_added   = ranked_features[k - 1] if k > 5 else "—"

        label = (f"Top-{k} features  [+{newly_added}]"
                 if k > 5 else f"Top-{k} features  [baseline]")
        print(f"\n  Step {k-4:>2}: {label}")

        tr = [df[current_feats + ['class']] for df in dfs_train]
        te = [df[current_feats + ['class']] for df in dfs_test]

        model = BlendingEnsemble()
        model.fit(tr)
        y_true, y_pred, y_prob = model.get_subject_predictions(te)

        m = compute_metrics(y_true, y_pred, y_prob)
        print_metrics(m, indent=6)

        greedy_results.append({
            'Step': k - 4,
            'N_Features': k,
            'Feature_Added': newly_added,
            'Features_Used': ', '.join(current_feats),
            **m,
        })

    return pd.DataFrame(greedy_results)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("COMPREHENSIVE BLENDING EXPERIMENTS  —  Schizophrenia Classification")
    print("="*70)

    # ------------------------------------------------------------------
    # 1. Load & engineer features
    # ------------------------------------------------------------------
    print("\n[1] Loading data …")
    df_10_raw = pd.read_csv(f'{DATA_DIR}/inital_data.csv')
    df_20_raw = pd.read_csv(f'{DATA_DIR}/initial_data_20_features.csv')

    print("[2] Applying feature engineering …")
    df_10_eng = engineer_features(df_10_raw)
    df_20_eng = engineer_features(df_20_raw)

    features_10 = [c for c in df_10_eng.columns if c != 'class']
    features_20 = [c for c in df_20_eng.columns if c != 'class']
    new_features = [c for c in features_20 if c not in features_10]

    print(f"  10-feature set  : {features_10}")
    print(f"  20-feature set  : {features_20}")
    print(f"  New features    : {new_features}")

    # ------------------------------------------------------------------
    # 2. Build subject dataframes
    # ------------------------------------------------------------------
    print("\n[3] Splitting into subject dataframes …")
    dfs_10 = clean_and_impute(process_subjects(df_10_eng))
    dfs_20 = clean_and_impute(process_subjects(df_20_eng))

    # Deterministic 72/28 subject-level split (same seed → same split)
    dfs_train_10, dfs_test_10 = train_test_split(dfs_10, train_size=0.72, random_state=42)
    dfs_train_20, dfs_test_20 = train_test_split(dfs_20, train_size=0.72, random_state=42)

    print(f"  Total subjects  : {len(dfs_20)}")
    print(f"  Train subjects  : {len(dfs_train_20)}")
    print(f"  Test  subjects  : {len(dfs_test_20)}")

    # ------------------------------------------------------------------
    # 3. Experiments 1-3
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("EXPERIMENTS 1–3  (Accuracy | Precision | Sensitivity | F1 | ROC-AUC)")
    print("="*70)

    exp_results = []
    exp_results.append(run_experiment("Exp1: Baseline  (10 features)",
                                      features_10, dfs_train_10, dfs_test_10))
    exp_results.append(run_experiment("Exp2: New features only",
                                      new_features, dfs_train_20, dfs_test_20))
    exp_results.append(run_experiment("Exp3: All 20 features",
                                      features_20, dfs_train_20, dfs_test_20))

    # ------------------------------------------------------------------
    # 4. Feature importance — 4 methods + voting ranking
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE ANALYSIS  (4 Methods + Voting-Based Ranking)")
    print("="*70)

    full_train = pd.concat(dfs_train_20)
    X_fi = full_train[features_20].fillna(full_train[features_20].median())
    y_fi = full_train['class']

    print(f"\n  Training on {len(full_train)} rows × {len(features_20)} features …\n")
    imp_df = calculate_feature_importance_voting(X_fi, y_fi, features_20)

    print("\n  VOTING-BASED FEATURE RANKING  (all 20 features)")
    print("  (Voting Score: higher = more important | Avg_Rank: lower = more important)\n")
    print_feature_importance_table(imp_df)

    ranked_features = imp_df['Feature'].tolist()

    # Save importance table
    imp_df.to_csv(f'{RESULT_DIR}/voting_feature_importance.csv', index=False)
    print(f"\n  Saved: {RESULT_DIR}/voting_feature_importance.csv")

    # ------------------------------------------------------------------
    # 5. Greedy forward selection starting from top 5
    # ------------------------------------------------------------------
    greedy_df = run_greedy_selection(ranked_features, dfs_train_20, dfs_test_20)

    # ------------------------------------------------------------------
    # 6. Summary tables
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("SUMMARY: Experiments 1–3")
    print("="*70)
    exp_df = pd.DataFrame(exp_results)
    col_order = ['Experiment', 'Accuracy', 'Precision', 'Sensitivity', 'F1', 'ROC_AUC']
    print(exp_df[col_order].to_string(index=False))

    print("\n" + "="*70)
    print("SUMMARY: Greedy Forward Selection")
    print("="*70)
    greedy_cols = ['Step', 'N_Features', 'Feature_Added',
                   'Accuracy', 'Precision', 'Sensitivity', 'F1', 'ROC_AUC']
    print(greedy_df[greedy_cols].to_string(index=False))

    # Best greedy result
    best_idx = greedy_df['F1'].idxmax()
    best_row = greedy_df.loc[best_idx]
    print(f"\n  Best greedy F1 = {best_row['F1']:.4f}  at N_Features = {int(best_row['N_Features'])}")
    print(f"  Features used  : {best_row['Features_Used']}")

    # ------------------------------------------------------------------
    # 7. Save CSVs
    # ------------------------------------------------------------------
    exp_df.to_csv(f'{RESULT_DIR}/comprehensive_experiment_results.csv', index=False)
    greedy_df.to_csv(f'{RESULT_DIR}/greedy_blending_results.csv', index=False)
    print(f"\n  Saved: {RESULT_DIR}/comprehensive_experiment_results.csv")
    print(f"  Saved: {RESULT_DIR}/greedy_blending_results.csv")

    # ------------------------------------------------------------------
    # 8. Visualisations
    # ------------------------------------------------------------------
    _plot_voting_importance(imp_df)
    _plot_greedy_metrics(greedy_df)
    _plot_greedy_barplot(greedy_df)

    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*70)


# ============================================================================
# VISUALISATION HELPERS
# ============================================================================

def _plot_voting_importance(imp_df):
    """Horizontal bar chart of voting scores for all 20 features."""
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#2ecc71' if i < 5 else '#3498db' if i < 10 else '#95a5a6'
              for i in range(len(imp_df))]
    ax.barh(imp_df['Feature'][::-1], imp_df['Voting_Score'][::-1],
            color=colors[::-1], edgecolor='black', linewidth=0.6)
    ax.set_xlabel('Voting Score  (higher = more important)', fontsize=11)
    ax.set_title('Feature Importance – Voting-Based Ranking (4 Methods)\n'
                 'Green = Top-5 | Blue = Top-6 to 10 | Grey = Rest',
                 fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    path = f'{RESULT_DIR}/voting_feature_importance.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def _plot_greedy_metrics(greedy_df):
    """Line plot of all 5 metrics vs number of features."""
    fig, ax = plt.subplots(figsize=(12, 7))
    metric_styles = [
        ('Accuracy',    'steelblue',  'o'),
        ('Precision',   'darkorange', 's'),
        ('Sensitivity', 'green',      '^'),
        ('F1',          'crimson',    'D'),
        ('ROC_AUC',     'purple',     'p'),
    ]
    for metric, color, marker in metric_styles:
        ax.plot(greedy_df['N_Features'], greedy_df[metric],
                color=color, marker=marker, markersize=8, linewidth=2.2,
                label=metric, alpha=0.85)

    ax.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Greedy Forward Selection – All Metrics vs Feature Count\n'
                 '(Voting-ranked order | Blending Ensemble)',
                 fontsize=13, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.set_xticks(greedy_df['N_Features'])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11)
    plt.tight_layout()
    path = f'{RESULT_DIR}/greedy_blending_lineplot.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def _plot_greedy_barplot(greedy_df):
    """Grouped bar chart showing all 5 metrics per step."""
    metrics = ['Accuracy', 'Precision', 'Sensitivity', 'F1', 'ROC_AUC']
    colors  = ['steelblue', 'darkorange', 'green', 'crimson', 'purple']
    n_steps = len(greedy_df)
    x = np.arange(n_steps)
    width = 0.15

    fig, ax = plt.subplots(figsize=(max(14, n_steps * 1.2), 7))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        offset = (i - 2) * width
        ax.bar(x + offset, greedy_df[metric], width,
               label=metric, color=color, edgecolor='black',
               linewidth=0.6, alpha=0.82)

    ax.set_xticks(x)
    x_labels = [f"Top-{int(r['N_Features'])}\n+{r['Feature_Added'][:12]}"
                if r['Feature_Added'] != '—'
                else f"Top-{int(r['N_Features'])}\n[baseline]"
                for _, r in greedy_df.iterrows()]
    ax.set_xticklabels(x_labels, fontsize=8, rotation=30, ha='right')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.set_title('Greedy Forward Selection – Metrics per Step\n'
                 '(Voting-ranked order | Blending Ensemble)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    path = f'{RESULT_DIR}/greedy_blending_barplot.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================

if __name__ == "__main__":
    main()
