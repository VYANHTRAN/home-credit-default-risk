# Home Credit Default Risk Pipeline

## Problem & Dataset

[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) is a Kaggle competition where the goal is to predict whether a loan applicant will default on repayment. The target population consists of people with little or no credit history - a group underserved by traditional scoring models - so the challenge is to build a reliable classifier from alternative data sources.

The dataset spans **9 tables** covering the full credit lifecycle:

| Table | Description |
|---|---|
| `application_train/test.csv` | Main table: one row per loan application with demographics, income, and credit details. Contains the binary `TARGET` (1 = default). |
| `bureau.csv` | Previous credits reported to the credit bureau |
| `bureau_balance.csv` | Monthly status history for each bureau credit |
| `previous_application.csv` | All prior Home Credit loan applications |
| `POS_CASH_balance.csv` | Monthly POS/cash loan balance snapshots |
| `credit_card_balance.csv` | Monthly credit card balance snapshots |
| `installments_payments.csv` | Repayment history for previous loans |

The training set contains ~307,000 applications and is heavily imbalanced (~8% default rate).

---

## Pipeline Overview

The pipeline runs two parallel tracks - **raw** and **processed** - and compares them via cross-validated AUC.

```
Raw CSVs
   │
   ├─ data_loader.py     → loads all 9 tables
   ├─ aggregation.py     → rolls auxiliary tables up to one row per SK_ID_CURR
   │
   ├── Pipeline A: Raw ──────────────────────────────────────────────────────┐
   │   preprocessing.py  → anomaly handling + categorical encoding           │
   │                                                                         ▼
   └── Pipeline B: Processed ──────────────────────────────────────────────┐ │
       preprocessing.py  → anomaly handling + categorical encoding         │ │
       feature_engineering.py → 10 feature groups (EXT_SOURCE interactions,│ │
                                affordability ratios, employment/age risk,  │ │
                                document/contact flags, credit enquiries…)  │ │
       preprocessing.py  → drop high-missing/high-corr columns,            │ │
                           log-transform skewed features                    ▼ ▼
                                                                        modelling.py
                                                                   LightGBM, 5-fold stratified CV
                                                                            │
                                                                        metrics.py
                                                                   ROC curves, feature importance,
                                                                   fold score plots, summary table
                                                                            │
                                                                   predictions/submission_*.csv
```

---

## Installation

**Requirements:** Python 3.10+

```bash
# 1. Clone the repo
git clone <repo-url>
cd home-credit-default-risk

# 2. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

Download the competition data from [Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data) and place all CSV files in a `data/` directory at the project root.

```
home-credit-default-risk/
├── data/
│   ├── application_train.csv
│   ├── application_test.csv
│   ├── bureau.csv
│   ├── bureau_balance.csv
│   ├── previous_application.csv
│   ├── POS_CASH_balance.csv
│   ├── credit_card_balance.csv
│   └── installments_payments.csv
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── aggregation.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── modelling.py
│   └── metrics.py
└── main.py
```

---

## Running

**Run both pipelines and compare (default):**
```bash
python main.py
```

**Run a single pipeline:**
```bash
python main.py --pipeline raw        # baseline only
python main.py --pipeline processed  # engineered features only
```

**Custom data or results directory:**
```bash
python main.py --data-dir /path/to/data --results-dir my_results
```

Outputs are written to:

- `results/` — ROC curve comparison, per-fold AUC box plots, feature importance charts
- `predictions/` — `submission_raw.csv` and/or `submission_processed.csv` (Kaggle format)
