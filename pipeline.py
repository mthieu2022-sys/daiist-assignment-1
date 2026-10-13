"""
Shared pipeline code imported by both train.py and app.py.

Keeping data loading, feature engineering, the chronological split, the
business-value function and the PyTorch model class in ONE place guarantees
that the app transforms data exactly the way training did.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "bank-additional-full.csv"
ARTIFACTS = ROOT / "artifacts"

SEED = 42

# --- Business assumptions (hypothetical, see REPORT.md) -----------------------
VALUE_PER_SALE = 50.0    # EUR: net margin the bank earns on one term deposit opened
COST_PER_CALL = 5.0      # EUR: agent time + telephony for one outbound call
ANNOYANCE_COST = 15.0    # EUR: expected goodwill cost of an UNWANTED call to an existing
                         #      client (complaints, opt-outs, small churn risk)

MONTHS = ["jan", "feb", "mar", "apr", "may", "jun",
          "jul", "aug", "sep", "oct", "nov", "dec"]

MACRO = ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]


# --- Loading -----------------------------------------------------------------
def load_raw() -> pd.DataFrame:
    """Load the CSV and rebuild the calendar year, which the file doesn't store.

    The rows are in chronological order (May 2008 -> Nov 2010, per the dataset
    documentation), so every time the month number goes DOWN, a new year began.
    """
    df = pd.read_csv(DATA_PATH, sep=";")
    month_num = df["month"].map({m: i for i, m in enumerate(MONTHS, 1)}).to_numpy()
    year = 2008 + np.concatenate([[0], np.cumsum(np.diff(month_num) < 0)])
    df["year"] = year
    df["period"] = pd.to_datetime(
        dict(year=df["year"], month=month_num, day=1)
    ).dt.strftime("%Y-%m")
    df["target"] = (df["y"] == "yes").astype(int)
    return df


# --- Chronological split -------------------------------------------------------
def split_by_time(df: pd.DataFrame):
    """Train on the past, tune on the recent past, test on the 'future'.

    train: May 2008 - Dec 2009   (everything the bank had when planning 2010)
    val:   Mar 2010 - Jun 2010   (used ONLY to pick C, learning rate, threshold)
    test:  Jul 2010 - Nov 2010   (touched once, at the very end)

    A random split would put 2010 calls in training and 2008 calls in test,
    i.e. let the model learn from the future. The conversion rate also moves
    from ~5% (2008) to ~50% (2010), so a random split would hide that drift.
    """
    train = df[df["period"] <= "2009-12"]
    val = df[(df["period"] >= "2010-01") & (df["period"] <= "2010-06")]
    test = df[df["period"] >= "2010-07"]
    return train.copy(), val.copy(), test.copy()


# --- Feature engineering -------------------------------------------------------
AGE_BINS = [0, 24, 34, 44, 54, 64, 200]
AGE_LABELS = ["<25", "25-34", "35-44", "45-54", "55-64", "65+"]

ENGINEERED_NUMERIC = ["log_campaign", "prev_contacted", "days_since_prev",
                      "log_previous"] + MACRO
ENGINEERED_CATEGORICAL = ["age_band", "job", "marital", "education", "default_status",
                          "housing", "loan", "contact", "day_of_week", "poutcome"]


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw columns into model-ready ones. Pure function: no fitting here.

    Every choice is explained in REPORT.md; the short version:
      * duration is NEVER used: it is only known after the call has ended.
      * month is NOT used: under a time split it encodes 'which year' (the
        conversion regime), not seasonality. The macro indicators carry the regime.
      * pdays=999 means 'never contacted' -> split into a flag + a real number.
      * campaign / previous are heavily right-skewed counts -> log1p.
      * age has a U-shaped effect (students & retirees convert most), which a
        linear model can't learn from raw age -> bands.
      * 'unknown' is kept as its own category: missingness is informative.
      * loan='unknown' is exactly the same 990 rows as housing='unknown', so that
        information is kept once (in housing); otherwise two columns are identical.
    """
    out = pd.DataFrame(index=df.index)
    out["log_campaign"] = np.log1p(df["campaign"])
    out["prev_contacted"] = (df["pdays"] != 999).astype(int)
    out["days_since_prev"] = np.where(df["pdays"] == 999, 0, df["pdays"])
    out["log_previous"] = np.log1p(df["previous"])
    for c in MACRO:
        out[c] = df[c]

    out["age_band"] = pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS).astype(str)
    out["education"] = df["education"].replace({"illiterate": "basic.4y"})  # 18 rows
    out["default_status"] = df["default"].replace({"yes": "unknown"})       # 3 rows
    out["loan"] = df["loan"].replace({"unknown": "no"})  # 'unknown' lives in housing
    for c in ["job", "marital", "housing", "contact", "day_of_week", "poutcome"]:
        out[c] = df[c]
    return out


def make_preprocessor(numeric, categorical) -> ColumnTransformer:
    """Scale numerics, one-hot categoricals. Fit on TRAIN only.

    drop='first' removes one level per categorical: with an intercept, a full
    one-hot set always sums to 1, which makes the problem flat in one
    direction. sklearn copes, but gradient descent crawls along that flat
    direction and the three implementations end up with different-looking
    (though equally good) coefficients.

    Scaling matters for two reasons here: L2 regularization should treat every
    coefficient on the same footing, and gradient descent in the two PyTorch
    versions converges far faster on standardized inputs.
    """
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore",
                                  sparse_output=False), categorical),
        ],
        verbose_feature_names_out=False,
    )


# --- Business value ----------------------------------------------------------------
def campaign_profit(y_true, y_prob, threshold, value=VALUE_PER_SALE,
                    cost=COST_PER_CALL, annoyance=ANNOYANCE_COST) -> float:
    """Net EUR of calling everyone whose predicted probability >= threshold.

    TP: called and subscribed  -> value - cost
    FP: called and declined    -> -(cost + annoyance)
    FN / TN: not called        -> 0 (FN is a missed sale, but no cash goes out)

    If p were calibrated, calling pays off when p*(value-cost) > (1-p)*(cost+annoyance),
    i.e. p > (cost+annoyance)/(value+annoyance) = 20/65 = 0.31 with the defaults.
    The model's probabilities are NOT calibrated for 2010 (trained mostly on a
    ~10% conversion regime), so the threshold is picked on validation instead.
    """
    y_true = np.asarray(y_true)
    call = np.asarray(y_prob) >= threshold
    tp = np.sum(call & (y_true == 1))
    fp = np.sum(call & (y_true == 0))
    return float(tp * (value - cost) - fp * (cost + annoyance))


# --- PyTorch models ------------------------------------------------------------------
class LogisticRegressionNet(torch.nn.Module):
    """Standard-workflow logistic regression: one linear layer.

    It outputs LOGITS; the sigmoid lives inside BCEWithLogitsLoss during training
    (numerically stable) and is applied explicitly at prediction time.
    """

    def __init__(self, n_features: int):
        super().__init__()
        self.linear = torch.nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


def manual_predict_proba(X: np.ndarray, w: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        z = torch.as_tensor(X, dtype=torch.float32) @ w + b
        return torch.sigmoid(z).numpy()


def standard_predict_proba(X: np.ndarray, model: LogisticRegressionNet) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.as_tensor(X, dtype=torch.float32))).numpy()
