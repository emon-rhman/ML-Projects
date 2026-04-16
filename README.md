# 🤖 Customer Churn Prediction – End-to-End ML Project

A production-quality machine learning project predicting **customer churn**,
covering every step from data generation to model deployment.

---

## 📁 Project Structure

```
ml_project/
├── config.py                  # Central config (paths, hyperparams, feature lists)
├── main.py                    # 🔁 Orchestrator – run this for full pipeline
├── predict.py                 # 🔮 Inference script (single or batch)
├── requirements.txt
│
├── data/
│   ├── generate_data.py       # Synthetic dataset generator (10,000 rows)
│   ├── raw_data.csv           # Generated on first run
│   └── processed_data.csv
│
├── src/
│   ├── data_loader.py         # Load + stratified train/val/test split
│   ├── preprocessing.py       # sklearn ColumnTransformer pipelines
│   ├── feature_engineering.py # Domain-specific feature creation
│   ├── models.py              # Model registry + hyperparameter grids
│   ├── training.py            # CV benchmark + RandomizedSearchCV tuning
│   ├── evaluation.py          # Metrics, threshold optimisation, all plots
│   └── utils.py               # Logger, model save/load, timer
│
├── notebooks/
│   └── eda.py                 # Exploratory data analysis (saves 5 plots)
│
├── models/                    # Saved .joblib pipelines
└── reports/                   # All CSV summaries + PNG plots
```

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Full end-to-end run
python main.py

# 3. Run EDA only
python notebooks/eda.py

# 4. Predict on new data
python predict.py --input new_customers.csv --output out.csv
```

---

## 🔬 Pipeline Steps

| # | Step | File |
|---|------|------|
| 1 | **Data Generation** | `data/generate_data.py` – 10k row synthetic dataset with realistic churn logic |
| 2 | **EDA** | `notebooks/eda.py` – 5 diagnostic plots saved to `reports/` |
| 3 | **Splitting** | `src/data_loader.py` – Stratified 70/10/20 split |
| 4 | **Preprocessing** | `src/preprocessing.py` – Imputation, scaling, one-hot encoding |
| 5 | **Feature Engineering** | `src/feature_engineering.py` – 7 new interaction features |
| 6 | **CV Benchmark** | `src/training.py` – 9 models compared via 5-fold stratified CV |
| 7 | **Hyperparameter Tuning** | RandomizedSearchCV on top-4 models |
| 8 | **Ensemble** | Soft-voting + Stacking meta-learner |
| 9 | **Evaluation** | ROC-AUC, PR-AUC, F1, Brier score; optimal threshold search |
| 10 | **Reporting** | Plots, JSON results, best model saved as `.joblib` |

---

## 🧠 Models Compared

- Logistic Regression
- Decision Tree
- Random Forest ⭐
- Extra Trees
- Gradient Boosting
- AdaBoost
- Support Vector Machine
- K-Nearest Neighbours
- Naïve Bayes
- XGBoost *(if installed)*
- LightGBM *(if installed)*
- **Soft-Voting Ensemble**
- **Stacking Ensemble** (LR meta-learner)

---

## 📊 Reports Generated

| File | Description |
|------|-------------|
| `cv_benchmark.csv` | 5-fold CV metrics for all base models |
| `test_results.csv` | Final test-set metrics (AUC, F1, PR-AUC, Brier) |
| `final_results.json` | Full experiment record |
| `01_churn_distribution.png` | Class balance pie + bar |
| `02_numerical_distributions.png` | Feature histograms by churn |
| `03_correlation_heatmap.png` | Feature correlation matrix |
| `04_categorical_churn_rates.png` | Churn rate per category |
| `05_boxplots.png` | Charges / tenure by churn |
| `roc_curves.png` | Multi-model ROC curves |
| `pr_curves.png` | Precision-recall curves |
| `calibration.png` | Calibration curves |
| `confusion_matrix.png` | Best model confusion matrix |
| `feature_importance.png` | Top-20 feature importances |
| `model_comparison.png` | Side-by-side metric bar chart |

---

## ⚙️ Configuration (`config.py`)

All key settings in one place:

```python
TEST_SIZE   = 0.20     # 20% test holdout
VAL_SIZE    = 0.10     # 10% validation
CV_FOLDS    = 5        # K-fold CV
N_ITER_RANDOM = 30     # RandomizedSearchCV iterations
SCORING_METRIC = "roc_auc"
```

---

## 🔮 Inference

```bash
# Batch CSV
python predict.py --input customers.csv --output preds.csv --threshold 0.45

# Single JSON record
python predict.py --json '{"tenure": 3, "monthly_charges": 90, "contract_type": "Month-to-month", ...}'
```

Output columns added:
- `churn_probability` – model confidence
- `churn_prediction`  – 0 or 1
- `risk_band`         – Low / Medium / High

---

## 🏗️ Extending

- **Add a new model**: register it in `src/models.py → get_base_models()`
- **Add features**: edit `src/feature_engineering.py`
- **Change target**: update `TARGET_COLUMN` in `config.py` and adjust feature lists
- **Switch dataset**: swap the CSV path in `config.py`
