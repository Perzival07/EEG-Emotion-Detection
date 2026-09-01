"""
Label-permutation significance test for the "AUC near chance" claim.

Sections 9-14 report ROC-AUC in the ~0.45-0.56 range under GroupKFold and call
it "near chance," but never formally tested whether the observed AUC differs
from a true empirical null. Khan et al. 2025 ("The Role of Review Process
Failures in Affective State Estimation," D:/EEG/papers_EEG/2508.02417v1.pdf)
show DEAP pipelines can look deceptively strong even on a label-randomized
"watermelon" control when evaluation is leaky -- the flip side, done properly
here, is to build a genuine null distribution *for this project's own already-
correct GroupKFold protocol* and see whether the observed AUC is actually
distinguishable from it.

Method: for each task, shuffle the labels (globally, independent of subject
grouping -- this is deliberate: it destroys any real feature/label relationship
while leaving the GroupKFold fold structure itself untouched), rerun the exact
same GroupKFold(5) pipeline used elsewhere in this repo, and record the mean
fold AUC. Repeat N_PERM times to build an empirical null distribution, then
p = (1 + #{null_auc >= observed_auc}) / (N_PERM + 1) (Ojala & Garriga 2010's
recommended +1 correction so p is never exactly 0).

Classifier: LogisticRegression only (not all 6 models). This is a documented,
pragmatic speed trade-off (same spirit as SVM_MAX_TRAIN_SAMPLES elsewhere in
this repo) -- section 9's ablation already showed inter-model AUC deltas on
the Full feature set are "well within one fold's standard deviation," so LR is
a fair, fast proxy for "is there any signal here at all," not a claim that LR
is the best model. N_PERM x 4 tasks x 5 folds x LogisticRegression fits are
parallelized across CPU cores with joblib.

Run: python scripts/track_a_significance_test.py
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
N_PERM = 300
TASK_ORDER = ["valence", "arousal", "dominance", "liking"]

CACHE_DIR = Path(r"D:/EEG/processed_cache")
RESULTS_DIR = Path(r"D:/EEG/results")
TABLES_DIR = RESULTS_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

ext = np.load(CACHE_DIR / "deap_features_ext_v1.npz")
X_full = np.concatenate([ext["X_de"], ext["X_hjorth"], ext["X_faa"], ext["X_conn"]], axis=1)
groups_all = ext["groups"]
y_by_task = {t: ext[f"y_{t}"] for t in TASK_ORDER}


def make_model():
    return LogisticRegression(max_iter=1000, random_state=SEED)


def mean_fold_auc(X, y, groups):
    gkf = GroupKFold(n_splits=5)
    aucs = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", make_model())])
        pipe.fit(X[train_idx], y[train_idx])
        y_prob = pipe.predict_proba(X[test_idx])[:, 1]
        aucs.append(roc_auc_score(y[test_idx], y_prob))
    return float(np.mean(aucs)) if aucs else np.nan


def one_permutation(X, y, groups, perm_seed):
    y_shuffled = np.random.RandomState(perm_seed).permutation(y)
    return mean_fold_auc(X, y_shuffled, groups)


def run_task(task):
    y = y_by_task[task]
    observed_auc = mean_fold_auc(X_full, y, groups_all)

    t0 = time.time()
    null_aucs = Parallel(n_jobs=-1)(
        delayed(one_permutation)(X_full, y, groups_all, SEED + i) for i in range(N_PERM)
    )
    null_aucs = np.array([a for a in null_aucs if not np.isnan(a)])
    elapsed = time.time() - t0

    p_value = (1 + np.sum(null_aucs >= observed_auc)) / (len(null_aucs) + 1)
    print(f"[{task}] observed_auc={observed_auc:.4f}  null: mean={null_aucs.mean():.4f} "
          f"std={null_aucs.std():.4f} min={null_aucs.min():.4f} max={null_aucs.max():.4f}  "
          f"p={p_value:.4f}  ({elapsed:.1f}s, n_perm={len(null_aucs)})")

    return {
        "task": task, "n_perm": len(null_aucs),
        "observed_auc": observed_auc,
        "null_auc_mean": null_aucs.mean(), "null_auc_std": null_aucs.std(),
        "null_auc_min": null_aucs.min(), "null_auc_max": null_aucs.max(),
        "z_score": (observed_auc - null_aucs.mean()) / (null_aucs.std() + 1e-12),
        "p_value": p_value,
        "elapsed_s": elapsed,
    }


if __name__ == "__main__":
    t0 = time.time()
    rows = [run_task(task) for task in TASK_ORDER]
    results = pd.DataFrame(rows)
    results.to_csv(TABLES_DIR / "track_a_significance_test.csv", index=False)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min. "
          f"Saved to {TABLES_DIR / 'track_a_significance_test.csv'}")
    print("\nSummary (alpha=0.05):")
    for _, r in results.iterrows():
        verdict = "SIGNIFICANT (but check effect size!)" if r.p_value < 0.05 else "not significant"
        print(f"  {r.task:<10} observed={r.observed_auc:.3f} null_mean={r.null_auc_mean:.3f} "
              f"z={r.z_score:+.2f} p={r.p_value:.4f} -> {verdict}")
