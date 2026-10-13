# Assignment 1 Report

- **Name**: Marcela Carla Torrelledó Hoyos
- **Student ID**: 19356
- **Email**: mth.ieu2022@student.ie.edu
- **Group**: BBADBA 5A

## Dataset

**UCI Bank Marketing, `bank-additional-full.csv`** (Moro, Cortez & Rita, 2014; UCI ML
Repository id 222), committed at `data/bank-additional-full.csv` (5.8 MB). The file is
byte-identical to the UCI version. UCI itself was blocked from my build environment, so I
downloaded it from a public GitHub mirror (`selva86/datasets`).

- **One row** = one phone contact made by a Portuguese bank during its term-deposit
  telemarketing campaigns, **May 2008 to November 2010**.
- **41,188 rows × 21 columns**: client attributes (age, job, marital, education, credit
  default, housing loan, personal loan), contact details (channel, month, weekday, number
  of contacts so far, days since the previous campaign, previous outcome), five monthly
  macro-economic indicators (employment variation rate, CPI, consumer confidence,
  Euribor 3m, number employed), the call `duration`, and the target `y` (subscribed?).
- **Why this one:** it has a real decision behind it, a leakage trap (`duration`), and
  strong drift over time. That makes the framing, the split and the features actual
  choices rather than defaults.

**Size note:** the brief suggests about 20k rows. I used the full 41k because the 10%
sample UCI distributes was drawn at random, which destroys the time order a
chronological split needs. The full file is still small (5.8 MB), and the whole training
run takes about 1.5 minutes on a laptop CPU.

The file has no year column, but rows are in date order. `load_raw()` rebuilds the year by
starting a new year every time the month number goes down (May 2008 → … → Nov 2010).

## Business / real-life framing

**Decision supported:** before each outbound call, should the call centre call this
client or skip them? The model gives P(subscribe | what is known *before* dialling). The
bank calls the client if that probability is at least a threshold.

The framing drove four concrete pipeline decisions:

1. **Target and features: only what is known before the call.** `duration` (call length)
   is the strongest predictor in the file, but it only exists *after* the call has ended,
   and a duration of 0 means `y = no` by construction. Using it would score well on paper
   and be useless in production. I excluded it, and kept it in the ablation only to
   measure the leak: validation ROC-AUC 0.70 → 0.81.
2. **Chronological split, not random.** The model would be built on past campaigns and
   used on future ones, so the split mirrors that:

   | split | period | rows | subscribe rate |
   |---|---|---|---|
   | train | 2008-05 → 2009-12 | 39,130 | 9.1% |
   | val | 2010-03 → 2010-06 | 879 | 54.2% |
   | test | 2010-07 → 2010-11 | 1,179 | 50.6% |

   A random split would mix 2010 calls into training, letting the model learn from the
   future. It would also hide the most important fact in this data: conversion rises from
   about 5% in 2008 to about 50% in 2010 (the dashboard's "Target over time" plot). The
   test window (the last ~1.2k contacts) matches the evaluation design of the original
   paper.
3. **The threshold is chosen by money, not by accuracy or 0.5.** Every call is a decision
   with an asymmetric payoff. With hypothetical but plausible numbers (all editable in
   the dashboard):
   - **TP** (called, subscribed): +€50 margin − €5 call cost = **+€45**
   - **FP** (called, declined): −€5 call cost − €15 goodwill cost of an unwanted call to
     an existing client (complaints, opt-outs) = **−€20**
   - **FN / TN** (not called): **€0** cash (an FN is a missed sale)

   The threshold that maximises this profit is picked **on validation**. It is then
   applied unchanged to test.
4. **Use validation for the threshold, not the theoretical break-even.** If the
   probabilities were calibrated, the break-even would be p = 20/65 = 0.31. They are not
   calibrated for 2010: the model learned mostly from a 9% conversion regime and
   under-predicts, as the calibration plot shows. Validation sits in the same recent
   regime as test, so the threshold it picks (0.15) corrects for that.

The right baseline is therefore **"call everyone"**, which is what the bank would do
without a model, not only the constant-prior classifier.

## Data preparation & feature engineering

All transformations are in `pipeline.engineer()` (a pure function, no fitting). The
scaler and one-hot encoder are fit on **train only**.

| What | Why |
|---|---|
| Drop `duration` | Post-call information (leak, see above). |
| Drop `month` | Under a time split, month mostly says *which year/regime* a call came from, not seasonality: rare months such as Mar/Sep/Oct/Dec happen almost only in the later high-conversion years. The month coefficients would encode "2009–10 vs 2008" from the training data. The macro indicators carry that regime information more honestly. Ablation: adding month gives +0.002 AUC, with a 95% interval from −0.010 to +0.016, i.e. noise. |
| `pdays = 999` → `prev_contacted` flag + `days_since_prev` (0 if never) | 999 is a sentinel, not a number. Scaled as-is, it would be the largest "days" value and distort the scale for the 4% who were contacted before. |
| `log1p(campaign)`, `log1p(previous)` | Heavily right-skewed counts (campaign goes up to 56). The log keeps a few extreme clients from dominating a linear model, and it helps the gradient-based PyTorch fits converge. |
| Age → bands (<25, 25–34, …, 65+) | The effect is U-shaped (students and retirees convert most; dashboard tab 2). A linear model on raw age can only learn a monotone effect. |
| Keep `"unknown"` as its own category | Missingness is informative. For example, `default = unknown` clients convert at 5.2% vs 12.9% for `no`. |
| `default = yes` (3 rows) → `unknown`; `education = illiterate` (18 rows) → `basic.4y` | Categories too small to estimate reliably. |
| `loan = unknown` → `no` | `loan = unknown` is **exactly the same 990 rows** as `housing = unknown`. Two identical columns make the problem singular, so the information is kept once, in housing. |
| One-hot with `drop="first"` | With an intercept, a full one-hot block always sums to 1, which creates a flat direction in the loss (see the comparison section for why this mattered). |
| `StandardScaler` on numerics | So that L2 regularisation treats coefficients equally, and so that gradient descent is well-conditioned. |

**Ablation** (validation ROC-AUC; each variant changes one thing and gets its own best C;
95% paired-bootstrap interval for the difference vs FINAL; produced by `train.py` and
shown in the dashboard):

| variant | val AUC | Δ vs final [95% CI] |
|---|---|---|
| **FINAL engineered set** (45 features) | **0.704** | — |
| raw columns, minimal prep, incl. month | 0.702 | −0.001 [−0.014, +0.015] |
| + month | 0.706 | +0.002 [−0.010, +0.016] |
| − macro indicators | 0.685 | −0.019 [−0.037, +0.002] |
| only euribor3m of the 5 macro columns | 0.703 | −0.001 [−0.009, +0.006] |
| + raw age & age² on top of bands | 0.702 | −0.001 [−0.007, +0.005] |
| + cellular × euribor interaction | 0.696 | −0.008 [−0.020, +0.003] |
| + recency weights (6-month half-life) | 0.698 | −0.006 [−0.014, +0.002] |
| train only on May08–May09 (older data) | 0.681 | **−0.022 [−0.037, −0.008]** |
| + duration (LEAK, not usable) | 0.814 | +0.111 (fake) |

**What didn't work:** I expected the engineered features to clearly beat the raw columns.
They don't. The difference is within noise. With only 879 validation rows, anything
under ±0.015 AUC can't be distinguished. A linear model on this data seems limited by how
much information is available before the call, not by how it's represented. Interactions,
age², recency weighting and dropping collinear macro columns also changed nothing
measurable.

What the feature work did buy:

- **Validity:** no leak, and no features that encode the calendar.
- **Robustness:** the raw set only matches at weak regularisation (C = 1), while the
  engineered set is stable across C.
- **Fewer features:** 45 vs 52.
- **Convergence:** removing the exactly collinear columns is what made the three
  implementations agree (next section).

The choices with measurable effects were the training window (including the recent
high-conversion months in training: +0.022) and keeping the macro indicators (+0.019).

## Modeling: three implementations, one model

**Model:** logistic regression, because the target is binary and the brief asks for one
simple model. All three versions minimise the **same objective**:

  mean log-loss + (λ/2)·‖w‖², with the bias not penalised

- **scikit-learn:** `LogisticRegression(C, lbfgs)`. sklearn minimises
  `C·Σ logloss + ½‖w‖²`. Dividing by C·n gives the per-sample form above with
  **λ = 1/(C·n)**, which is how C is translated for PyTorch. C was tuned on validation
  AUC over {0.001, …, 10}; C = 0.01 → λ = 2.56e-3.
- **Manual PyTorch:**
  - Raw tensors `w`, `b`, starting at zero.
  - Loss written by hand as `softplus(z) − y·z`, which is the same as −[y·log σ + (1−y)·log(1−σ)]
    but numerically stable.
  - `loss.backward()`, then `w -= lr·w.grad` under `torch.no_grad()`, then zero the
    gradients.
  - Full-batch gradient descent, 5,000 steps.
  - Learning rate chosen from {0.1, 0.5, 1, 2, 4} as the one reaching the lowest training
    objective without diverging: lr = 2 (lr = 4 diverged). Because the problem is convex,
    the "best" lr is simply the one that gets to the shared optimum, not a validation
    choice.
- **Standard PyTorch:**
  - `nn.Module` with one `nn.Linear`, `BCEWithLogitsLoss`, `DataLoader` mini-batches of
    512, `torch.optim.Adam`.
  - The L2 term is set through `weight_decay = λ` on the **weight parameter group only**
    (Adam's weight_decay adds λw to the gradient, which is exactly the gradient of
    (λ/2)‖w‖²).
  - Cosine learning-rate annealing over 60 epochs, so the mini-batch noise dies out.
  - lr from {0.001, 0.01, 0.05} by the same rule: 0.05.

**Results on the test set** (Jul–Nov 2010, 1,179 calls; threshold picked on validation):

| model | ROC-AUC | PR-AUC | log-loss | threshold | calls | precision | recall | profit |
|---|---|---|---|---|---|---|---|---|
| baseline: constant prior (p = 0.091) | 0.500 | 0.506 | 1.260 | 0.50 | 0 | – | 0.000 | €0 |
| baseline: call everyone | 0.500 | 0.506 | – | 0.00 | 1,179 | 0.506 | 1.000 | €15,225 |
| scikit-learn | 0.744 | 0.732 | 0.651 | 0.15 | 1,122 | 0.524 | 0.985 | **€15,780** |
| manual PyTorch | 0.744 | 0.732 | 0.650 | 0.15 | 1,122 | 0.524 | 0.985 | **€15,780** |
| standard PyTorch | 0.744 | 0.732 | 0.650 | 0.15 | 1,123 | 0.524 | 0.985 | **€15,760** |

**Do the three agree? Yes, to optimisation precision:**

| pair | max \|Δp\| | mean \|Δp\| | max \|Δcoef\| | same decision |
|---|---|---|---|---|
| sklearn vs manual | 0.0005 | 0.0002 | 0.0018 | 100% |
| sklearn vs standard | 0.0081 | 0.0013 | 0.0033 | 99.9% (1 client) |
| manual vs standard | 0.0081 | 0.0012 | 0.0034 | 99.9% |

The final objective values are 0.265152 (sklearn), 0.265152 (manual) and 0.265153
(standard). Full-batch gradient descent reaches sklearn's optimum almost exactly (to
float32 precision). Adam on mini-batches is about 1e-6 above it, which is the residual
mini-batch noise. That gap is why standard PyTorch makes a different decision for one
client and earns €20 less.

**This was not true on my first attempt, and the investigation is the most useful thing I
learned.** In my first version I one-hot encoded every level (no `drop="first"`) and kept
both `housing = unknown` and `loan = unknown`. sklearn and both PyTorch versions then had
almost identical *predictions* but visibly different *coefficients*: up to 0.35 apart, for
example `poutcome_nonexistent` +0.16 in sklearn vs −0.19 in Adam. The reason:

- Each full one-hot block sums to 1, the same as the intercept column, and the two
  `unknown` columns were identical.
- So there were directions in weight space where the predictions don't change at all.
- Only the tiny L2 term curves the loss in those directions. lbfgs (second-order
  information) finds that minimum. Gradient descent barely moves along them in a few
  thousand steps.

I confirmed this from the eigenvalues of XᵀX/n: one was 0, with an eigenvector of
±0.71 on exactly those two columns. Dropping one level per category and de-duplicating
the `unknown` column removed the flat directions. The coefficient gap fell from 0.35 to
0.003. Training time: sklearn 0.6 s, manual 5 s, standard 14 s.

**The model vs the baselines:**

- **Ranking quality:** the model clearly beats the constant prior (AUC 0.74 vs 0.50).
- **Business value:** under my default costs it only beats "call everyone" by €555
  (+3.6%), by skipping 57 low-probability clients. When half of everyone called says yes,
  calling everyone is a strong policy.
- **The value grows with the cost of an unwanted call** (`train.py` section 6; threshold
  re-picked on validation each time):

| goodwill cost per unwanted call | threshold | calls | model profit | call everyone |
|---|---|---|---|---|
| €0 | 0.01 | 1,179 | €23,955 | €23,955 |
| €15 (default) | 0.15 | 1,122 | €15,780 | €15,225 |
| €30 | 0.25 | 813 | €12,025 | €6,495 |
| €50 | 0.29 | 654 | €7,630 | −€5,145 |

So the honest conclusion is that this model is worth deploying when unwanted calls are
expensive, or in a low-conversion regime like 2008, where calling everyone loses money.
It is barely worth it if calls are cheap and conversion is around 50%.

## Limitations & next steps

1. **Strong drift and extrapolation.** 93% of the training rows come from May 2008 – May
   2009, when conversion was about 7%. In the test months, `nr.employed`
   (4,964–4,992) lies entirely *below* the training range (minimum 5,017.5), so the model
   extrapolates linearly. Its probabilities are too low (calibration plot). *Next:* retrain monthly on a rolling
   window. Recalibrate on the most recent months, for example with a Platt/intercept
   shift fit on validation; that is cheap and the right fix for a shift in the base rate.
2. **The split was revised after looking at test results.** My first split trained only
   up to May 2009 and tested on all of 2010 (test AUC about 0.64; the model added nothing
   over calling everyone). I then moved the training cut-off to include 2009 H2, which is
   what a bank planning 2010 would actually have. The revision is defensible, but test
   was not completely untouched. *Next:* confirm on genuinely new campaign data, or use
   rolling-origin evaluation over several cut-offs.
3. **Validation is small (879 rows).** The threshold and C are noisy, and AUC differences
   below about ±0.015 aren't detectable (see the bootstrap intervals). *Next:* rolling
   validation over several windows, and pick the threshold from a smoothed profit curve.
4. **The cost numbers are assumptions.** €50 / €5 / €15 are plausible but invented, and
   the conclusion depends on them (sensitivity table). *Next:* get the real margin per
   deposit, the call cost, and opt-out/complaint rates from the bank. The dashboard
   already takes them as inputs.
5. **Rows are calls, not clients.** The same client can appear several times within a
   campaign, and there is no client ID to de-duplicate or group by. *Next:* with IDs, use
   group-aware splitting and predict "will this client subscribe in this campaign", not
   "on this call".
6. **The linear model's ceiling.** Everything I engineered added nothing measurable,
   which suggests limited pre-call information rather than missing transforms. *Next:* a
   tree ensemble would show whether non-linear interactions exist. Better still would be
   richer pre-call data (account balance, product holdings, channel history), as in the
   original paper.

## Generative AI use disclosure

I used **Claude (Anthropic)** as a coding and writing assistant for this assignment. It:

- proposed the dataset, among options I asked for;
- wrote the first versions of `pipeline.py`, `train.py`, `app.py` and this report;
- ran the experiments that the numbers above come from.

I reviewed every design decision: the leakage exclusion, the time split, the cost
function, the λ = 1/(C·n) mapping, and the collinearity investigation. I can explain each
of them without the code.
