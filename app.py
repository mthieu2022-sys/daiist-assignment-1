"""
Assignment 1 - Gradio dashboard (run with: uv run python main.py app).

Loads the artifacts written by train.py and NEVER trains anything: the three
models are read from disk, the saved preprocessor transforms the data, and all
predictions below come from those loaded models.
"""

import json

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from pipeline import (AGE_LABELS, ANNOYANCE_COST, ARTIFACTS, COST_PER_CALL, ENGINEERED_CATEGORICAL,
                      ENGINEERED_NUMERIC, VALUE_PER_SALE, LogisticRegressionNet,
                      campaign_profit, engineer, load_raw, manual_predict_proba,
                      split_by_time, standard_predict_proba)

# ---------------------------------------------------------------- load artifacts
if not (ARTIFACTS / "metadata.json").exists():
    raise SystemExit("artifacts/ not found - run `uv run python main.py train` first.")

meta = json.loads((ARTIFACTS / "metadata.json").read_text())
preprocessor = joblib.load(ARTIFACTS / "preprocessor.joblib")
sk_model = joblib.load(ARTIFACTS / "sklearn_logreg.joblib")
manual_state = torch.load(ARTIFACTS / "manual_torch.pt")
std_model = LogisticRegressionNet(len(meta["feature_names"]))
std_model.load_state_dict(torch.load(ARTIFACTS / "standard_torch.pt"))
coef = pd.read_csv(ARTIFACTS / "coefficients.csv")
ablation = pd.read_csv(ARTIFACTS / "ablation.csv")

MODELS = ["scikit-learn", "manual PyTorch", "standard PyTorch"]
COLORS = {"scikit-learn": "#2a6fdb", "manual PyTorch": "#e07a1f", "standard PyTorch": "#2a9d55"}

# ---------------------------------------------------------------- data + predictions
df = load_raw()
train_df, val_df, test_df = split_by_time(df)
df["split"] = np.select([df.index.isin(train_df.index), df.index.isin(val_df.index)],
                        ["train", "val"], "test")
SPLITS = {"val": val_df, "test": test_df}
cols = ENGINEERED_NUMERIC + ENGINEERED_CATEGORICAL


def predict_all(part):
    X = preprocessor.transform(engineer(part)[cols]).astype(np.float32)
    return {
        "scikit-learn": sk_model.predict_proba(X)[:, 1],
        "manual PyTorch": manual_predict_proba(X, manual_state["w"], manual_state["b"]),
        "standard PyTorch": standard_predict_proba(X, std_model),
    }


PRED = {name: predict_all(part) for name, part in SPLITS.items()}
Y = {name: part["target"].to_numpy() for name, part in SPLITS.items()}


# ---------------------------------------------------------------- tab 1: comparison
def metrics_table():
    y, rows = Y["test"], []
    for m in MODELS:
        p, t = PRED["test"][m], meta["thresholds"][m]
        pred = p >= t
        rows.append({"model": m, "ROC-AUC": roc_auc_score(y, p),
                     "PR-AUC": average_precision_score(y, p),
                     "threshold (from val)": t, "calls": int(pred.sum()),
                     "precision": (pred & (y == 1)).sum() / max(pred.sum(), 1),
                     "recall": (pred & (y == 1)).sum() / (y == 1).sum(),
                     "profit EUR": campaign_profit(y, p, t),
                     "fit seconds": meta["fit_seconds"][m]})
    rows.append({"model": "baseline: constant prior", "ROC-AUC": 0.5, "PR-AUC": y.mean(),
                 "threshold (from val)": 0.5, "calls": 0, "precision": 0.0, "recall": 0.0,
                 "profit EUR": 0.0, "fit seconds": 0.0})
    rows.append({"model": "baseline: call everyone", "ROC-AUC": 0.5, "PR-AUC": y.mean(),
                 "threshold (from val)": 0.0, "calls": len(y), "precision": y.mean(),
                 "recall": 1.0, "profit EUR": campaign_profit(y, np.ones(len(y)), 0.0),
                 "fit seconds": 0.0})
    return pd.DataFrame(rows).round(4)


def calibration_plot(split):
    """Predicted vs actual for a binary target: bin predicted probabilities and
    compare each bin's average prediction with the share that actually said yes."""
    y = Y[split]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect calibration",
                             line=dict(color="grey", dash="dash")))
    for m in MODELS:
        p = PRED[split][m]
        bins = pd.qcut(p, 10, duplicates="drop")
        g = pd.DataFrame({"p": p, "y": y}).groupby(bins, observed=True).mean()
        fig.add_trace(go.Scatter(x=g["p"], y=g["y"], mode="lines+markers", name=m,
                                 line=dict(color=COLORS[m])))
    fig.update_layout(title=f"Predicted probability vs actual subscribe rate ({split}, deciles)",
                      xaxis_title="mean predicted probability",
                      yaxis_title="actual share who subscribed",
                      xaxis_range=[0, 1], yaxis_range=[0, 1], height=450)
    return fig


def agreement_plot(split):
    p = PRED[split]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="identical",
                             line=dict(color="grey", dash="dash")))
    for m in ["manual PyTorch", "standard PyTorch"]:
        fig.add_trace(go.Scatter(x=p["scikit-learn"], y=p[m], mode="markers", name=m,
                                 marker=dict(color=COLORS[m], size=5, opacity=0.5)))
    fig.update_layout(title=f"Per-client probability: PyTorch vs scikit-learn ({split})",
                      xaxis_title="scikit-learn probability", yaxis_title="PyTorch probability",
                      height=450)
    return fig


def prob_by_outcome_plot(split, model):
    d = pd.DataFrame({"predicted probability": PRED[split][model],
                      "actual": np.where(Y[split] == 1, "subscribed", "declined")})
    fig = px.histogram(d, x="predicted probability", color="actual", barmode="overlay",
                       nbins=40, opacity=0.6,
                       color_discrete_map={"subscribed": "#2a9d55", "declined": "#c0392b"})
    fig.update_layout(title=f"{model}: predicted probability by actual outcome ({split})",
                      height=400)
    return fig


def coefficient_plot():
    c = coef.copy()
    c["order"] = c["scikit-learn"].abs()
    c = c.sort_values("order", ascending=True).tail(25)
    fig = go.Figure()
    for m in MODELS:
        fig.add_trace(go.Bar(y=c["feature"], x=c[m], name=m, orientation="h",
                             marker_color=COLORS[m]))
    fig.update_layout(barmode="group", height=750,
                      title="Top 25 coefficients (standardised features): three implementations",
                      xaxis_title="coefficient (log-odds per unit / vs reference level)")
    return fig


def loss_plot():
    """How far each PyTorch run is from the scikit-learn optimum, epoch by epoch.

    Two panels because an 'epoch' means different things: one full-batch step
    for the manual loop vs one pass over ~77 mini-batches for the standard one.
    """
    opt = meta["objective"]["scikit-learn"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=list(meta["loss_history"]))
    for i, (m, hist) in enumerate(meta["loss_history"].items(), start=1):
        h = pd.DataFrame(hist)
        fig.add_trace(go.Scatter(x=h["epoch"], y=np.maximum(h["loss"] - opt, 1e-8), name=m,
                                 line=dict(color=COLORS[m])), row=1, col=i)
        fig.update_xaxes(title_text="epoch", row=1, col=i)
        fig.update_yaxes(type="log", row=1, col=i)
    fig.update_yaxes(title_text="objective minus sklearn optimum", row=1, col=1)
    fig.update_layout(title="Convergence to the shared optimum (log scale)", height=400,
                      showlegend=False)
    return fig


# ---------------------------------------------------------------- tab 2: data
RAW_FEATURES = ["age", "job", "marital", "education", "default", "housing", "loan", "contact",
                "day_of_week", "campaign", "pdays", "previous", "poutcome", "emp.var.rate",
                "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed", "duration"]
ENG_VIEW = engineer(df).add_prefix("eng: ")
EXPLORE = pd.concat([df, ENG_VIEW], axis=1)
EXPLORE["outcome"] = np.where(df["target"] == 1, "subscribed", "declined")
FEATURE_CHOICES = RAW_FEATURES + list(ENG_VIEW.columns)


def feature_plot(feature, split):
    d = EXPLORE if split == "all" else EXPLORE[EXPLORE["split"] == split]
    if not pd.api.types.is_numeric_dtype(d[feature]) or d[feature].nunique() <= 12:
        rate = d.groupby(feature)["target"].agg(["mean", "size"]).reset_index()
        if feature == "eng: age_band":
            rate[feature] = pd.Categorical(rate[feature], categories=AGE_LABELS, ordered=True)
            rate = rate.sort_values(feature)
        fig = go.Figure(go.Bar(x=rate[feature].astype(str), y=rate["size"], name="clients",
                               marker_color="#9aa7b8"))
        fig.add_trace(go.Scatter(x=rate[feature].astype(str), y=rate["mean"], yaxis="y2",
                                 name="subscribe rate", mode="lines+markers",
                                 line=dict(color="#2a9d55")))
        fig.update_layout(yaxis=dict(title="clients"),
                          yaxis2=dict(title="subscribe rate", overlaying="y", side="right",
                                      range=[0, 1]))
    else:
        fig = px.histogram(d, x=feature, color="outcome", barmode="overlay", nbins=50,
                           opacity=0.6,
                           color_discrete_map={"subscribed": "#2a9d55", "declined": "#c0392b"})
    note = "  [duration = LEAK: only known after the call, never used]" if feature == "duration" else ""
    fig.update_layout(title=f"{feature} ({split}){note}", height=450)
    return fig


def target_over_time_plot():
    g = df.groupby(["period", "split"]).agg(calls=("target", "size"),
                                            rate=("target", "mean")).reset_index()
    fig = go.Figure()
    for s, col in [("train", "#9aa7b8"), ("val", "#e07a1f"), ("test", "#2a6fdb")]:
        gs = g[g["split"] == s]
        fig.add_trace(go.Bar(x=gs["period"], y=gs["calls"], name=f"calls ({s})",
                             marker_color=col))
    fig.add_trace(go.Scatter(x=g["period"], y=g["rate"], yaxis="y2", name="subscribe rate",
                             mode="lines+markers", line=dict(color="#2a9d55")))
    fig.update_layout(title="Target over time: calls per month and share who subscribed",
                      yaxis=dict(title="calls"),
                      yaxis2=dict(title="subscribe rate", overlaying="y", side="right",
                                  range=[0, 1]),
                      xaxis=dict(type="category"), height=450)
    return fig


# ---------------------------------------------------------------- tab 3: threshold
def threshold_view(model, split, threshold, value, cost, annoyance):
    y, p = Y[split], PRED[split][model]
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    profit = campaign_profit(y, p, threshold, value, cost, annoyance)
    call_all = campaign_profit(y, p, 0.0, value, cost, annoyance)

    cm = go.Figure(go.Heatmap(
        z=[[tn, fp], [fn, tp]], x=["predicted: don't call", "predicted: call"],
        y=["actual: declined", "actual: subscribed"], colorscale="Blues", showscale=False,
        text=[[f"TN {tn}<br>EUR 0", f"FP {fp}<br>-EUR {fp * (cost + annoyance):,.0f}"],
              [f"FN {fn}<br>missed EUR {fn * (value - cost):,.0f}",
               f"TP {tp}<br>+EUR {tp * (value - cost):,.0f}"]],
        texttemplate="%{text}", textfont=dict(size=16)))
    cm.update_layout(title=f"Confusion matrix - {model}, {split}, threshold {threshold:.2f}",
                     yaxis_autorange="reversed", height=400)

    ts = np.round(np.arange(0.0, 1.0, 0.01), 2)
    curve = go.Figure(go.Scatter(x=ts, y=[campaign_profit(y, p, t, value, cost, annoyance)
                                          for t in ts], name="model", line=dict(color=COLORS[model])))
    curve.add_hline(y=call_all, line_dash="dash", line_color="grey",
                    annotation_text="call everyone")
    curve.add_vline(x=threshold, line_color="black")
    curve.update_layout(title="Campaign profit vs threshold", xaxis_title="threshold",
                        yaxis_title="profit (EUR)", height=400)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    breakeven = (cost + annoyance) / (value + annoyance)
    text = (f"### Profit at this threshold: **EUR {profit:,.0f}**\n"
            f"- Call everyone: EUR {call_all:,.0f} -> model is "
            f"**EUR {profit - call_all:+,.0f}** vs that\n"
            f"- Calls made: {tp + fp} of {len(y)} | precision {precision:.1%} | "
            f"recall {recall:.1%} | accuracy {(tp + tn) / len(y):.1%}\n"
            f"- Break-even probability if the model were calibrated: {breakeven:.2f}\n"
            f"- Threshold picked on validation in training: "
            f"{meta['thresholds'][model]:.2f} (at the default costs)")
    return cm, curve, text


# ---------------------------------------------------------------- layout
split_info = meta["split"]
intro = f"""
# Who should the bank call? - term-deposit campaign model
Logistic regression trained three ways (scikit-learn, manual PyTorch loop, standard
`nn.Module` + `torch.optim`) on a **chronological** split:
train {split_info['train']['from']} to {split_info['train']['to']} ({split_info['train']['rows']:,} calls),
val {split_info['val']['from']} to {split_info['val']['to']} ({split_info['val']['rows']:,}),
test {split_info['test']['from']} to {split_info['test']['to']} ({split_info['test']['rows']:,}).
Everything here comes from models loaded from `artifacts/`; nothing is retrained.
"""

with gr.Blocks(title="Bank campaign model dashboard") as demo:
    gr.Markdown(intro)

    with gr.Tab("1. Compare the three models"):
        gr.Markdown("Test-set results. Thresholds were picked on validation to maximise profit.")
        gr.Dataframe(value=metrics_table(), interactive=False)
        cmp_split = gr.Radio(["test", "val"], value="test", label="data split")
        with gr.Row():
            calib = gr.Plot(value=calibration_plot("test"))
            agree = gr.Plot(value=agreement_plot("test"))
        hist_model = gr.Dropdown(MODELS, value="scikit-learn", label="model")
        hist = gr.Plot(value=prob_by_outcome_plot("test", "scikit-learn"))
        cmp_split.change(lambda s, m: (calibration_plot(s), agreement_plot(s),
                                       prob_by_outcome_plot(s, m)),
                         [cmp_split, hist_model], [calib, agree, hist])
        hist_model.change(prob_by_outcome_plot, [cmp_split, hist_model], hist)
        with gr.Row():
            gr.Plot(value=coefficient_plot())
            gr.Plot(value=loss_plot())

    with gr.Tab("2. Features & target"):
        gr.Plot(value=target_over_time_plot())
        with gr.Row():
            feat = gr.Dropdown(FEATURE_CHOICES, value="eng: age_band", label="feature")
            feat_split = gr.Radio(["all", "train", "val", "test"], value="all", label="split")
        feat_fig = gr.Plot(value=feature_plot("eng: age_band", "all"))
        feat.change(feature_plot, [feat, feat_split], feat_fig)
        feat_split.change(feature_plot, [feat, feat_split], feat_fig)
        gr.Markdown("### Feature ablation (validation ROC-AUC, 95% paired-bootstrap interval "
                    "for the difference vs the final set)")
        gr.Dataframe(value=ablation.round(4), interactive=False)

    with gr.Tab("3. Decision threshold & business cost"):
        with gr.Row():
            th_model = gr.Dropdown(MODELS, value="scikit-learn", label="model")
            th_split = gr.Radio(["test", "val"], value="test", label="data split")
        threshold = gr.Slider(0.0, 0.99, value=meta["thresholds"]["scikit-learn"], step=0.01,
                              label="decision threshold: call the client if P(subscribe) >= this")
        with gr.Row():
            value_in = gr.Number(VALUE_PER_SALE, label="value of one deposit sold (EUR)")
            cost_in = gr.Number(COST_PER_CALL, label="cost of one call (EUR)")
            annoy_in = gr.Number(ANNOYANCE_COST, label="goodwill cost of an unwanted call (EUR)")
        summary = gr.Markdown()
        with gr.Row():
            cm_fig = gr.Plot()
            curve_fig = gr.Plot()
        inputs = [th_model, th_split, threshold, value_in, cost_in, annoy_in]
        outputs = [cm_fig, curve_fig, summary]
        for comp in inputs:
            comp.change(threshold_view, inputs, outputs)
        demo.load(threshold_view, inputs, outputs)

if __name__ == "__main__":
    demo.launch(inbrowser=True)
