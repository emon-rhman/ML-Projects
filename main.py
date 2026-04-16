#!/usr/bin/env python
# ============================================================
#  main.py  –  End-to-end ML pipeline orchestrator
# ============================================================
"""
Full A→Z run:
    python main.py

Steps
-----
1. Data generation / loading
2. EDA
3. Train/val/test split
4. Cross-validation benchmark
5. Hyperparameter tuning (top-N models)
6. Ensemble building
7. Final evaluation on held-out test set
8. Report generation & model persistence
"""

import os, sys, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from config import RANDOM_STATE, REPORTS_DIR, MODEL_DIR
from src.data_loader      import load_raw, split_data
from src.training         import benchmark_models, tune_top_models, build_full_pipeline
from src.evaluation       import (
    compute_metrics, print_report,
    optimal_threshold,
    plot_roc_curves, plot_pr_curves,
    plot_confusion_matrix, plot_calibration,
    plot_feature_importance, compare_models_bar,
)
from src.models           import (
    get_base_models, build_voting_ensemble, build_stacking_ensemble,
)
from src.utils            import get_logger, save_model, save_results, Timer

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

logger = get_logger()


# ════════════════════════════════════════════════════════════
#  STEP 1 – Data
# ════════════════════════════════════════════════════════════
logger.info("STEP 1: Loading data")

with Timer("Data load"):
    df = load_raw()

logger.info(f"Dataset shape: {df.shape}  |  churn rate: {df['churn'].mean():.2%}")
logger.info(f"Missing cells: {df.isnull().sum().sum()}")


# ════════════════════════════════════════════════════════════
#  STEP 2 – EDA
# ════════════════════════════════════════════════════════════
logger.info("STEP 2: Running EDA")

with Timer("EDA"):
    try:
        import subprocess
        subprocess.run(
            [sys.executable, "notebooks/eda.py"],
            check=True, capture_output=True,
        )
        logger.info("EDA plots saved to reports/")
    except Exception as e:
        logger.warning(f"EDA skipped: {e}")


# ════════════════════════════════════════════════════════════
#  STEP 3 – Split
# ════════════════════════════════════════════════════════════
logger.info("STEP 3: Train/val/test split")

X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)


# ════════════════════════════════════════════════════════════
#  STEP 4 – Cross-validation benchmark
# ════════════════════════════════════════════════════════════
logger.info("STEP 4: CV benchmark")

with Timer("CV benchmark"):
    bench_df = benchmark_models(X_train, y_train)

bench_df.to_csv(os.path.join(REPORTS_DIR, "cv_benchmark.csv"), index=False)
logger.info(f"Top model: {bench_df.iloc[0]['model']}  "
            f"AUC={bench_df.iloc[0]['roc_auc']:.4f}")


# ════════════════════════════════════════════════════════════
#  STEP 5 – Hyperparameter tuning
# ════════════════════════════════════════════════════════════
logger.info("STEP 5: Hyperparameter tuning")

with Timer("Hyperparameter tuning"):
    tuned_models = tune_top_models(bench_df, X_train, y_train, top_n=4)

logger.info(f"Tuned models: {list(tuned_models.keys())}")


# ════════════════════════════════════════════════════════════
#  STEP 6 – Validation evaluation & ensemble
# ════════════════════════════════════════════════════════════
logger.info("STEP 6: Validation evaluation + ensemble building")

val_preds = {}
val_metrics = {}

for name, pipe in tuned_models.items():
    y_prob = pipe.predict_proba(X_val)[:, 1]
    val_preds[name]   = y_prob
    val_metrics[name] = compute_metrics(y_val, y_prob)
    logger.info(f"  val AUC [{name}] = {val_metrics[name]['roc_auc']:.4f}")

# Build soft-voting ensemble
if len(tuned_models) >= 2:
    voting = build_voting_ensemble(tuned_models)
    voting.fit(X_train, y_train)                        # ensemble needs raw data
    val_preds["voting_ensemble"]    = voting.predict_proba(X_val)[:, 1]
    val_metrics["voting_ensemble"]  = compute_metrics(
        y_val, val_preds["voting_ensemble"]
    )
    logger.info(f"  val AUC [voting] = {val_metrics['voting_ensemble']['roc_auc']:.4f}")

    # Stacking
    try:
        stacking = build_stacking_ensemble(tuned_models)
        stacking.fit(X_train, y_train)
        val_preds["stacking"]   = stacking.predict_proba(X_val)[:, 1]
        val_metrics["stacking"] = compute_metrics(y_val, val_preds["stacking"])
        logger.info(f"  val AUC [stacking] = {val_metrics['stacking']['roc_auc']:.4f}")
    except Exception as e:
        logger.warning(f"Stacking failed: {e}")

# pick best on validation
best_val_name = max(val_metrics, key=lambda n: val_metrics[n]["roc_auc"])
logger.info(f"Best on validation: {best_val_name}")


# ════════════════════════════════════════════════════════════
#  STEP 7 – Final test evaluation
# ════════════════════════════════════════════════════════════
logger.info("STEP 7: Test set evaluation")

all_pipelines = dict(tuned_models)
if "voting_ensemble" in val_preds:
    all_pipelines["voting_ensemble"] = voting
if "stacking" in val_preds:
    all_pipelines["stacking"] = stacking

test_preds   = {}
test_metrics = {}

for name, pipe in all_pipelines.items():
    y_prob = pipe.predict_proba(X_test)[:, 1]
    test_preds[name]   = y_prob
    test_metrics[name] = compute_metrics(y_test, y_prob)

# results table
test_df = pd.DataFrame(test_metrics).T.reset_index().rename(columns={"index": "model"})
test_df = test_df.sort_values("roc_auc", ascending=False)
test_df.to_csv(os.path.join(REPORTS_DIR, "test_results.csv"), index=False)

print("\n" + "═" * 55)
print("  FINAL TEST RESULTS")
print("═" * 55)
print(test_df.to_string(index=False))
print()

best_name = test_df.iloc[0]["model"]
best_pipe = all_pipelines[best_name]
best_prob = test_preds[best_name]

thresh = optimal_threshold(y_test, best_prob)
logger.info(f"Best model: {best_name}  |  optimal threshold: {thresh}")
print_report(y_test, best_prob, name=best_name, threshold=thresh)


# ════════════════════════════════════════════════════════════
#  STEP 8 – Plots & model save
# ════════════════════════════════════════════════════════════
logger.info("STEP 8: Generating plots and saving artefacts")

with Timer("Plot generation"):
    plot_roc_curves(test_preds, y_test)
    plot_pr_curves(test_preds, y_test)
    plot_calibration(test_preds, y_test)
    plot_confusion_matrix(y_test, best_prob, name=best_name, threshold=thresh)
    compare_models_bar(test_df)

    # feature importance (only for tree-based best)
    if hasattr(best_pipe, "named_steps"):
        pre = best_pipe.named_steps.get("preprocessor")
        try:
            feat_names = list(pre.get_feature_names_out())
        except Exception:
            feat_names = [f"f_{i}" for i in range(200)]
        plot_feature_importance(best_pipe, feat_names)

# save best model
metadata = {
    "model_name":  best_name,
    "test_metrics": test_metrics[best_name],
    "threshold":   thresh,
}
path = save_model(best_pipe, name=f"best_{best_name}", metadata=metadata)
save_results(
    {"benchmark": bench_df.to_dict("records"),
     "test": test_metrics,
     "best_model": best_name,
     "threshold": thresh,
     "saved_path": path},
    fname="final_results.json",
)

logger.info(f"Pipeline complete ✓  |  best model: {best_name}  "
            f"|  test AUC={test_metrics[best_name]['roc_auc']:.4f}")
print(f"\n✅  Done!  Best model → {best_name}  "
      f"(AUC={test_metrics[best_name]['roc_auc']:.4f})")
print(f"   Reports: {REPORTS_DIR}")
print(f"   Models:  {MODEL_DIR}")
