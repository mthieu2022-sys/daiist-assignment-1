"""
Assignment 1 - training pipeline (run with: uv run python main.py train).

Problem: a Portuguese bank's call centre decides WHICH clients to phone in a
term-deposit campaign. We predict P(subscribe | what is known BEFORE the call)
with logistic regression, trained three ways on the same chronological split:

  1. scikit-learn LogisticRegression
  2. a from-scratch PyTorch loop (raw tensors + autograd + manual update)
  3. the standard torch.nn.Module + torch.optim + DataLoader workflow

All three minimise the SAME objective (mean log-loss + L2 on the weights),
so they should land on (almost) the same coefficients. Everything the app
needs is written to artifacts/; the app never trains anything.
"""

import json
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             log_loss, precision_score, recall_score, roc_auc_score)
from torch.utils.data import DataLoader, TensorDataset

from pipeline import (ANNOYANCE_COST, ARTIFACTS, COST_PER_CALL, ENGINEERED_CATEGORICAL, ENGINEERED_NUMERIC,
                      SEED, VALUE_PER_SALE, LogisticRegressionNet, campaign_profit, engineer,
                      load_raw, make_preprocessor, manual_predict_proba, split_by_time,
                      standard_predict_proba)

np.random.seed(SEED)
torch.manual_seed(SEED)
ARTIFACTS.mkdir(exist_ok=True)
THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2)


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def ranking_metrics(y, p):
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7))),
        "brier": float(brier_score_loss(y, p)),
    }


def best_threshold(y, p):
    """Pick the threshold that maximises campaign profit on VALIDATION data."""
    profits = [campaign_profit(y, p, t) for t in THRESHOLDS]
    return float(THRESHOLDS[int(np.argmax(profits))])


def decision_metrics(y, p, t):
    pred = (p >= t).astype(int)
    return {
        "threshold": t,
        "profit_eur": campaign_profit(y, p, t),
        "calls": int(pred.sum()),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
    }


# =============================================================================
section("1. Data and chronological split")
df = load_raw()
train_df, val_df, test_df = split_by_time(df)
for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
    print(f"{name:5s} {part['period'].min()} -> {part['period'].max()}  "
          f"rows={len(part):6d}  subscribe-rate={part['target'].mean():.3f}")
y_train, y_val, y_test = (d["target"].to_numpy() for d in (train_df, val_df, test_df))


# =============================================================================
section("2. Feature ablation (scikit-learn, each variant at its own best C, VALIDATION only)")
# Each variant changes ONE thing relative to the final feature set, and each gets
# its own best C from the same grid, so no variant loses just because it needs
# a different amount of regularisation. The paired
# bootstrap gives a 95% interval for the AUC difference vs FINAL on the same
# validation rows, so we can tell real gains from noise (val has only 879 rows).
RAW_NUMERIC = ["age", "campaign", "pdays", "previous", "emp.var.rate", "cons.price.idx",
               "cons.conf.idx", "euribor3m", "nr.employed"]
RAW_CATEGORICAL = ["job", "marital", "education", "default", "housing", "loan", "contact",
                   "month", "day_of_week", "poutcome"]
MACRO_COLS = ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
NO_MACRO = [c for c in ENGINEERED_NUMERIC if c not in MACRO_COLS]


def recency_weights(d, half_life_months=6):
    """Down-weight old rows: a row 6 months older counts half as much."""
    age = (pd.to_datetime(d["period"].max()) - pd.to_datetime(d["period"])).dt.days / 30.4
    return (0.5 ** (age / half_life_months)).to_numpy()


def eng_plus(**extra):
    """engineer() plus extra columns computed from the raw frame."""
    def build(d):
        e = engineer(d)
        for name, fn in extra.items():
            e[name] = fn(d)
        return e
    return build


raw = lambda d: d  # noqa: E731
N, CAT = ENGINEERED_NUMERIC, ENGINEERED_CATEGORICAL
# label: (frame builder, numeric cols, categorical cols, options)
VARIANTS = {
    "FINAL engineered set": (engineer, N, CAT, {}),
    "raw columns, minimal prep (incl. month)": (raw, RAW_NUMERIC, RAW_CATEGORICAL, {}),
    "+ month": (eng_plus(month=lambda d: d["month"]), N, CAT + ["month"], {}),
    "- macro indicators": (engineer, NO_MACRO, CAT, {}),
    "only euribor3m of the 5 macro cols": (
        engineer, NO_MACRO + ["euribor3m"], CAT, {}),
    "+ raw age & age^2 (on top of bands)": (
        eng_plus(age=lambda d: d["age"], age_sq=lambda d: (d["age"] - 40) ** 2),
        N + ["age", "age_sq"], CAT, {}),
    "+ cellular x euribor interaction": (
        eng_plus(cell_x_euribor=lambda d: (d["contact"] == "cellular") * d["euribor3m"]),
        N + ["cell_x_euribor"], CAT, {}),
    "+ recency weights (6-month half-life)": (engineer, N, CAT, {"recency": True}),
    "train window May08-May09 only (old data)": (engineer, N, CAT, {"old_only": True}),
    "+ duration  [LEAK - not usable]": (
        eng_plus(log_duration=lambda d: np.log1p(d["duration"])), N + ["log_duration"], CAT, {}),
}
rng = np.random.default_rng(SEED)
boot_idx = [rng.integers(0, len(y_val), len(y_val)) for _ in range(300)]
ablation, val_preds = [], {}
for label, (build, num, cat, opt) in VARIANTS.items():
    tr = train_df[train_df["period"] <= "2009-05"] if opt.get("old_only") else train_df
    pre = make_preprocessor(num, cat)
    Xa_tr = pre.fit_transform(build(tr)[num + cat])
    Xa_va = pre.transform(build(val_df)[num + cat])
    sw = recency_weights(tr) if opt.get("recency") else None
    best = (-1, None, None)
    for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
        m = LogisticRegression(C=C, max_iter=5000).fit(Xa_tr, tr["target"], sample_weight=sw)
        pc = m.predict_proba(Xa_va)[:, 1]
        auc = roc_auc_score(y_val, pc)
        if auc > best[0]:
            best = (auc, C, pc)
    _, best_c_variant, p = best
    val_preds[label] = p
    base = val_preds["FINAL engineered set"]
    diffs = [roc_auc_score(y_val[i], p[i]) - roc_auc_score(y_val[i], base[i]) for i in boot_idx]
    row = {"variant": label, "n_features": Xa_tr.shape[1], "best_C": best_c_variant,
           "val_roc_auc": roc_auc_score(y_val, p), "val_pr_auc": average_precision_score(y_val, p),
           "delta_auc_vs_final": roc_auc_score(y_val, p) - roc_auc_score(y_val, base),
           "delta_ci_low": float(np.percentile(diffs, 2.5)),
           "delta_ci_high": float(np.percentile(diffs, 97.5))}
    ablation.append(row)
    print(f"{label:42s} n={row['n_features']:3d} C={best_c_variant:<5}  AUC={row['val_roc_auc']:.3f}  "
          f"PR-AUC={row['val_pr_auc']:.3f}  dAUC={row['delta_auc_vs_final']:+.3f} "
          f"[{row['delta_ci_low']:+.3f}, {row['delta_ci_high']:+.3f}]")


# =============================================================================
section("3. Final preprocessing (fit on TRAIN only)")
preprocessor = make_preprocessor(ENGINEERED_NUMERIC, ENGINEERED_CATEGORICAL)
cols = ENGINEERED_NUMERIC + ENGINEERED_CATEGORICAL
X_train = preprocessor.fit_transform(engineer(train_df)[cols]).astype(np.float32)
X_val = preprocessor.transform(engineer(val_df)[cols]).astype(np.float32)
X_test = preprocessor.transform(engineer(test_df)[cols]).astype(np.float32)
feature_names = list(preprocessor.get_feature_names_out())
n, d = X_train.shape
print(f"X_train {X_train.shape}, X_val {X_val.shape}, X_test {X_test.shape}")


# =============================================================================
section("4. Naive baselines")
prior = float(y_train.mean())
baseline_val = np.full(len(y_val), prior)
baseline_test = np.full(len(y_test), prior)
print(f"Constant prior p={prior:.3f}: ranks nobody above anybody (ROC-AUC 0.5); at the "
      "0.5 default threshold it calls nobody -> EUR 0.")
print(f"'Call everyone' profit on test: EUR {campaign_profit(y_test, baseline_test, 0.0):,.0f}")


# =============================================================================
section("5a. scikit-learn LogisticRegression (tune C on validation ROC-AUC)")
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
c_scores = {}
for C in C_GRID:
    m = LogisticRegression(C=C, max_iter=5000).fit(X_train, y_train)
    c_scores[C] = roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])
    print(f"  C={C:<6} val ROC-AUC={c_scores[C]:.4f}")
best_C = max(c_scores, key=c_scores.get)
t0 = time.time()
sk_model = LogisticRegression(C=best_C, max_iter=10000, tol=1e-8).fit(X_train, y_train)
sk_time = time.time() - t0
print(f"-> chosen C={best_C}  (fit in {sk_time:.2f}s)")

# sklearn minimises   C * sum(logloss) + 0.5 * ||w||^2   (bias not penalised).
# Dividing by C*n gives the per-sample form used by the PyTorch versions:
#     mean(logloss) + (lam / 2) * ||w||^2     with  lam = 1 / (C * n)
lam = 1.0 / (best_C * n)
print(f"Equivalent PyTorch L2 strength: lam = 1/(C*n) = {lam:.3e}")

Xt_train = torch.tensor(X_train)
yt_train = torch.tensor(y_train, dtype=torch.float32)


def objective(w, b):
    """The shared objective, evaluated full-batch (used to compare optimisers)."""
    with torch.no_grad():
        z = Xt_train @ w + b
        return float((torch.nn.functional.softplus(z) - yt_train * z).mean()
                     + lam / 2 * (w ** 2).sum())


sk_w = torch.tensor(sk_model.coef_[0], dtype=torch.float32)
sk_b = torch.tensor(sk_model.intercept_[0], dtype=torch.float32)
sk_obj = objective(sk_w, sk_b)
print(f"sklearn objective value: {sk_obj:.6f}")


# =============================================================================
section("5b. Manual PyTorch loop (full-batch gradient descent)")
MANUAL_EPOCHS = 5000


def train_manual(lr, epochs=MANUAL_EPOCHS, log_every=25):
    torch.manual_seed(SEED)
    w = torch.zeros(d, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    history = []
    for epoch in range(epochs):
        z = Xt_train @ w + b                                   # forward: logits
        # log-loss written out by hand: -[y*log(s) + (1-y)*log(1-s)]
        # == softplus(z) - y*z  (same thing, but numerically stable)
        data_loss = (torch.nn.functional.softplus(z) - yt_train * z).mean()
        loss = data_loss + lam / 2 * (w ** 2).sum()            # + L2 on weights only
        loss.backward()                                        # autograd fills .grad
        with torch.no_grad():                                  # manual update step
            w -= lr * w.grad
            b -= lr * b.grad
            w.grad.zero_()
            b.grad.zero_()
        if epoch % log_every == 0 or epoch == epochs - 1:
            history.append({"epoch": epoch, "loss": loss.item()})
        if not np.isfinite(loss.item()):
            break
    return w.detach(), b.detach(), history


manual_runs = {}
for lr in [0.1, 0.5, 1.0, 2.0, 4.0]:
    t0 = time.time()
    w, b, hist = train_manual(lr)
    obj = objective(w, b)
    manual_runs[lr] = (w, b, hist, obj, time.time() - t0)
    print(f"  lr={lr:<4} final objective={obj:.6f}  (gap to sklearn {obj - sk_obj:+.2e})")
# Convex problem -> the 'best' learning rate is simply the one that gets closest
# to the optimum within the epoch budget without diverging.
best_lr_manual = min((k for k in manual_runs if np.isfinite(manual_runs[k][3])),
                     key=lambda k: manual_runs[k][3])
man_w, man_b, man_hist, man_obj, man_time = manual_runs[best_lr_manual]
print(f"-> chosen lr={best_lr_manual}")


# =============================================================================
section("5c. Standard PyTorch workflow (nn.Module + DataLoader + torch.optim.Adam)")
STD_EPOCHS = 60
BATCH = 512


def train_standard(lr, epochs=STD_EPOCHS):
    torch.manual_seed(SEED)
    model = LogisticRegressionNet(d)
    torch.nn.init.zeros_(model.linear.weight)
    torch.nn.init.zeros_(model.linear.bias)
    # Adam's weight_decay adds lam*w to the gradient == gradient of (lam/2)||w||^2.
    # Separate param groups so the BIAS is not penalised (same as sklearn).
    optimizer = torch.optim.Adam([
        {"params": [model.linear.weight], "weight_decay": lam},
        {"params": [model.linear.bias], "weight_decay": 0.0},
    ], lr=lr)
    # Mini-batch gradients are noisy; annealing the lr to ~0 lets it settle.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    g = torch.Generator().manual_seed(SEED)
    loader = DataLoader(TensorDataset(Xt_train, yt_train), batch_size=BATCH, shuffle=True,
                        generator=g)
    history = []
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()
        history.append({"epoch": epoch,
                        "loss": objective(model.linear.weight[0].detach(),
                                          model.linear.bias.detach())})
    return model, history


std_runs = {}
for lr in [0.001, 0.01, 0.05]:
    t0 = time.time()
    model, hist = train_standard(lr)
    obj = objective(model.linear.weight[0].detach(), model.linear.bias.detach())
    std_runs[lr] = (model, hist, obj, time.time() - t0)
    print(f"  lr={lr:<5} final objective={obj:.6f}  (gap to sklearn {obj - sk_obj:+.2e})")
best_lr_std = min(std_runs, key=lambda k: std_runs[k][2])
std_model, std_hist, std_obj, std_time = std_runs[best_lr_std]
print(f"-> chosen lr={best_lr_std}")


# =============================================================================
section("6. Evaluation: validation picks the threshold, test is reported")
probs = {
    "baseline (constant prior)": {"val": baseline_val, "test": baseline_test},
    "scikit-learn": {"val": sk_model.predict_proba(X_val)[:, 1],
                     "test": sk_model.predict_proba(X_test)[:, 1]},
    "manual PyTorch": {"val": manual_predict_proba(X_val, man_w, man_b),
                       "test": manual_predict_proba(X_test, man_w, man_b)},
    "standard PyTorch": {"val": standard_predict_proba(X_val, std_model),
                         "test": standard_predict_proba(X_test, std_model)},
}
results = {}
for name, p in probs.items():
    t = 0.5 if name.startswith("baseline") else best_threshold(y_val, p["val"])
    results[name] = {"val": ranking_metrics(y_val, p["val"]) if not name.startswith("baseline")
                     else {"roc_auc": 0.5, "pr_auc": float(y_val.mean()),
                           "log_loss": float(log_loss(y_val, p["val"])),
                           "brier": float(brier_score_loss(y_val, p["val"]))},
                     "test": (ranking_metrics(y_test, p["test"]) if not name.startswith("baseline")
                              else {"roc_auc": 0.5, "pr_auc": float(y_test.mean()),
                                    "log_loss": float(log_loss(y_test, p["test"])),
                                    "brier": float(brier_score_loss(y_test, p["test"]))})}
    results[name]["test"].update(decision_metrics(y_test, p["test"], t))
results["call everyone"] = {"test": {"profit_eur": campaign_profit(y_test, baseline_test, 0.0),
                                     "calls": int(len(y_test)), "threshold": 0.0,
                                     "precision": float(y_test.mean()), "recall": 1.0,
                                     "accuracy": float(y_test.mean())}}

table = pd.DataFrame({k: v["test"] for k, v in results.items()}).T
cols_show = ["roc_auc", "pr_auc", "log_loss", "brier", "threshold", "calls",
             "precision", "recall", "accuracy", "profit_eur"]
print(table[cols_show].to_string(float_format=lambda x: f"{x:,.3f}"))


# How much the model is worth depends on the cost assumptions: when calls are
# cheap and half of 2010's clients say yes, 'call everyone' is hard to beat.
print("\nSensitivity to the annoyance cost (scikit-learn, threshold re-picked on val):")
sensitivity = []
for annoy in [0.0, 15.0, 30.0, 50.0]:
    pv, pt = probs["scikit-learn"]["val"], probs["scikit-learn"]["test"]
    prof_v = [campaign_profit(y_val, pv, t, annoyance=annoy) for t in THRESHOLDS]
    t_star = float(THRESHOLDS[int(np.argmax(prof_v))])
    row = {"annoyance_cost": annoy, "threshold": t_star,
           "model_profit": campaign_profit(y_test, pt, t_star, annoyance=annoy),
           "call_everyone_profit": campaign_profit(y_test, pt, 0.0, annoyance=annoy),
           "calls": int((pt >= t_star).sum())}
    sensitivity.append(row)
    print(f"  annoyance EUR {annoy:>4.0f}: threshold {t_star:.2f}, calls {row['calls']:4d}, "
          f"model EUR {row['model_profit']:>8,.0f} vs call-everyone EUR "
          f"{row['call_everyone_profit']:>8,.0f}")


# =============================================================================
section("7. Do the three implementations agree?")
coef = pd.DataFrame({"feature": feature_names,
                     "scikit-learn": sk_model.coef_[0],
                     "manual PyTorch": man_w.numpy(),
                     "standard PyTorch": std_model.linear.weight[0].detach().numpy()})
agreement = {}
pairs = [("scikit-learn", "manual PyTorch"), ("scikit-learn", "standard PyTorch"),
         ("manual PyTorch", "standard PyTorch")]
for a, b_ in pairs:
    dp = np.abs(probs[a]["test"] - probs[b_]["test"])
    dc = np.abs(coef[a] - coef[b_])
    same = np.mean((probs[a]["test"] >= results["scikit-learn"]["test"]["threshold"]) ==
                   (probs[b_]["test"] >= results["scikit-learn"]["test"]["threshold"]))
    agreement[f"{a} vs {b_}"] = {"max_abs_prob_diff": float(dp.max()),
                                 "mean_abs_prob_diff": float(dp.mean()),
                                 "max_abs_coef_diff": float(dc.max()),
                                 "same_decision_rate": float(same)}
print(pd.DataFrame(agreement).T.to_string(float_format=lambda x: f"{x:.5f}"))
print("\nObjective values (lower = closer to the shared optimum):")
print(f"  scikit-learn {sk_obj:.6f} | manual {man_obj:.6f} | standard {std_obj:.6f}")
worst = coef.assign(gap=(coef["scikit-learn"] - coef["standard PyTorch"]).abs())
print("Largest coefficient gaps (sklearn vs standard):")
print(worst.nlargest(5, "gap").to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# =============================================================================
section("8. Saving artifacts")
joblib.dump(preprocessor, ARTIFACTS / "preprocessor.joblib")
joblib.dump(sk_model, ARTIFACTS / "sklearn_logreg.joblib")
torch.save({"w": man_w, "b": man_b}, ARTIFACTS / "manual_torch.pt")
torch.save(std_model.state_dict(), ARTIFACTS / "standard_torch.pt")
coef.to_csv(ARTIFACTS / "coefficients.csv", index=False)
pd.DataFrame(ablation).to_csv(ARTIFACTS / "ablation.csv", index=False)

metadata = {
    "split": {name: {"from": part["period"].min(), "to": part["period"].max(),
                     "rows": int(len(part)), "positive_rate": float(part["target"].mean())}
              for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]},
    "business": {"value_per_sale": VALUE_PER_SALE, "cost_per_call": COST_PER_CALL,
                 "annoyance_cost": ANNOYANCE_COST},
    "hyperparameters": {"C": best_C, "lam": lam, "C_grid_val_auc": c_scores,
                        "manual_lr": best_lr_manual, "manual_epochs": MANUAL_EPOCHS,
                        "standard_lr": best_lr_std, "standard_epochs": STD_EPOCHS,
                        "standard_batch": BATCH},
    "objective": {"scikit-learn": sk_obj, "manual PyTorch": man_obj, "standard PyTorch": std_obj},
    "fit_seconds": {"scikit-learn": sk_time, "manual PyTorch": man_time,
                    "standard PyTorch": std_time},
    "thresholds": {k: v["test"]["threshold"] for k, v in results.items()},
    "results": results,
    "agreement": agreement,
    "cost_sensitivity": sensitivity,
    "feature_names": feature_names,
    "loss_history": {"manual PyTorch": man_hist, "standard PyTorch": std_hist},
}
with open(ARTIFACTS / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2, default=float)
print(f"Saved to {ARTIFACTS}/: " + ", ".join(sorted(p.name for p in ARTIFACTS.iterdir())))
