import matplotlib.pyplot as plt

# Expected files and names 
TABLE_FILES = {
    "train":        "application_train.csv",
    "test":         "application_test.csv",
    "bureau":       "bureau.csv",
    "bureau_bal":   "bureau_balance.csv",
    "prev_app":     "previous_application.csv",
    "pos_cash":     "POS_CASH_balance.csv",
    "cc_bal":       "credit_card_balance.csv",
    "installments": "installments_payments.csv",
}

# Stakeholder value used by Home Credit for "not employed / pensioner / unemployed"
DAYS_EMPLOYED_ANOMALY = 365243
 
# Threshold for hish missing rate 
HIGH_MISSING_THRESHOLD = 50.0
 
# Threshold for high correlation rate 
HIGH_CORR_THRESHOLD = 0.8
 
# Skewness threshold above which a numeric column is log-transformed
SKEW_THRESHOLD = 1.0

# Parameters for LightGBM model 
DEFAULT_PARAMS: dict = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "n_estimators": 5000,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "class_weight": "balanced",   # handles the imbalanced classes in Home Credit
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

EARLY_STOPPING_ROUNDS = 100
VERBOSE_EVAL = 200

# Default style 
PALETTE = {
    "raw":       "#4C72B0",   # baseline model
    "processed": "#DD8452",   # engineered model
}
FALLBACK_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Directory for results
RESULTS_DIR = "results"