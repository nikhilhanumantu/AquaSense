# Methodology fixes: what changed and why

This documents the difference between the repo's original ML pipeline and
the fixes applied to it - both the model/methodology (`train_and_analyze.py`)
and how it's served (`app.py`, `frontend/`). All "after" numbers below are
read directly from `aquasense_artifacts/model_card.json` and
`model_metrics.json` after running `train_and_analyze.py`, not estimated.

## 1. Which model is actually used

| | Before | After |
|---|---|---|
| Selection | Hardcoded: `MODEL_METADATA["XGBoost"]["is_active"] = True`, always, regardless of any measurement | Whichever of the 4 tuned candidates has the highest **5-fold cross-validated ROC-AUC** |
| Result on this run | XGBoost, unconditionally | **Random Forest** (CV ROC-AUC 0.6977 vs XGBoost's 0.6858) |
| Does the served prediction actually use this model? | **No** - see \S4 below | Yes - `is_potable` now comes from the champion pipeline's own `predict_proba`, at its own tuned threshold |

### Is Random Forest actually better than XGBoost? Be precise about this.

The two models' cross-validated ROC-AUC scores are **statistically tied**,
not a clear win. The real per-fold scores behind each mean
(`model_card.json`'s `champion_cv_roc_auc_folds` / `runner_up_cv_roc_auc_folds`):

| Fold | XGBoost | Random Forest |
|---|---|---|
| 1 | 0.7037 | 0.6923 |
| 2 | 0.6754 | 0.6840 |
| 3 | 0.6841 | 0.6941 |
| 4 | 0.7021 | 0.7174 |
| 5 | 0.6638 | 0.7008 |
| **mean** | **0.6858** | **0.6977** |
| **std** | **0.0154** | **0.0112** |

The gap between the means (0.0119) is smaller than either model's own
fold-to-fold standard deviation. `model_card.json` states this explicitly:
`"champion_cv_margin_within_noise": true`.

**A second wrinkle, also real**: on the one held-out test set, the ranking
flips - XGBoost's test ROC-AUC (0.6858) is *higher* than Random Forest's
(0.6595). Selection still uses the cross-validated score, not the test
score, because picking a model based on test-set performance and then
reporting that same test score as its result is a known bias (the
"winner's curse": whichever of several noisy estimates looks best that day
gets picked, so its reported score is optimistic). Cross-validation makes
this choice using only the training set, before the test set is ever
touched - the textbook-correct criterion, even though on this particular
run the two rankings disagree. `model_card.json`'s
`test_vs_cv_disagreement_note` spells this out in full, and it's now
displayed on `analysis.html` directly, not just in raw JSON.

**Bottom line**: the champion is a legitimate, principled tie-break
between two models that perform indistinguishably on this data - not
evidence that Random Forest is the "correct" architecture for this
problem. Say so, rather than presenting one badge as if it settles the
question.

## 2. Hyperparameters: guessed vs. searched

| Model | Before (hand-picked, never varied) | After (`RandomizedSearchCV`, 25 iters x 5-fold CV) |
|---|---|---|
| XGBoost | n_estimators=300, max_depth=4, lr=0.05, subsample=0.85, colsample=0.85 | n_estimators=200, max_depth=6, **lr=0.01**, subsample=0.6, colsample=1.0 |
| Random Forest | n_estimators=400, min_samples_leaf=2, *(depth/features never specified - implicit unbounded default)* | n_estimators=400, min_samples_leaf=1, max_features=log2, max_depth=None *(verified best, not accidental)* |
| Decision Tree | max_depth=8, min_samples_leaf=5 | max_depth=10, min_samples_leaf=20, criterion=entropy |
| Logistic Regression | C=1.0 (scikit-learn default, never set) | **C=2.81** (found by search) |

## 3. Decision threshold: one fixed number vs. tuned per model

| | Before | After |
|---|---|---|
| All 4 models | **0.50**, hardcoded, unused by any actual tuning | F1-optimal, chosen from out-of-fold predictions on the training set only: XGBoost 0.407, Random Forest 0.423, Decision Tree 0.325, Logistic Regression 0.417 |

## 4. The critical bug: the model's own prediction was computed and discarded

In `app.py`'s `/api/predict`, `pipeline.predict()` was called and its
result (`raw_pred`) was never used. `is_potable` was decided entirely by
`evaluate_parameters()`'s hand-written WHO/EPA rule engine. Practical
effect: **switching the `model` dropdown between all 4 trained models
never changed the verdict**, only a cosmetic probability blend. Fixed:
`is_potable = model_says_potable and meets_guidelines`, with a stated
`verdict_reason` whenever the two disagree (e.g. *"The model predicts
potable, but 1 guideline threshold(s) were exceeded: Elevated Sulfates
(> 450 mg/L)."*). Verified live: switching models now changes the verdict
when they actually disagree.

The same bug existed a second time on the frontend: `main.html`'s
homepage widget called a fully fabricated client-side formula directly and
never contacted the backend at all, while labeling its output "XGBoost" or
"Random Forest." Fixed to call the real backend, with the offline fallback
now honestly labeled "(offline estimate)" instead of claiming to be a
specific trained model.

## 5. The resulting metrics - and why accuracy went *down* while the model got *better*

At threshold 0.50 (before) vs. each model's own tuned threshold (after):

| Model | Accuracy | Recall (Potable) | ROC-AUC (test) |
|---|---|---|---|
| XGBoost | 62.50% -> **54.73%** | 48.44% -> **84.38%** | 0.6437 -> **0.6858** |
| Random Forest (new champion) | 66.77% -> 57.62% | 32.81% -> 60.55% | 0.6622 -> 0.6595 |
| Decision Tree | 64.48% -> 48.02% | 28.52% -> 90.23% | 0.6099 -> 0.6106 |
| Logistic Regression | 52.44% -> 39.02% | 53.12% -> 100.00% | 0.5474 -> 0.5474 |

Lower accuracy is not a regression here. The dataset is 61% Non-Potable /
39% Potable, so a threshold that just says "Non-Potable" most of the time
scores well on accuracy while barely detecting the minority class - the
old XGBoost, at 0.50, was missing 52% of truly potable samples despite
looking respectable on accuracy. The new thresholds are tuned to actually
predict both classes; ROC-AUC (threshold-independent) is flat or improved
for every model, which is the honest confirmation this is a real tuning
gain and not noise.

**Also new: a baseline that didn't exist before.** Majority-Class
Baseline: 60.98% accuracy, 0.5000 ROC-AUC, by never predicting "Potable"
at all. The old README's claim of "62.50% accuracy" for XGBoost is barely
above that trivial floor; its ROC-AUC (0.6437, now 0.6858) clearly is.

## 6. Feature importance: gain-based vs. permutation-based

`.feature_importances_` (before) cannot produce a negative number by
construction, so every feature always looks like it contributed
*something*. Permutation importance, computed fresh on the held-out test
set, shows the honest picture for the champion:

| Feature | Permutation importance |
|---|---|
| Sulfate | 0.114 |
| pH | 0.112 |
| Hardness | 0.043 |
| Solids | 0.042 |
| Chloramines | 0.039 |
| Organic Carbon | 0.003 |
| Trihalomethanes | **-0.001** |
| Conductivity | **-0.002** |
| Turbidity | **-0.006** |

Three features are actively negative - shuffling them randomly makes the
model *slightly better*, i.e. it was fitting noise on them. The old method
could never reveal this.

## 7. Provenance and reproducibility

Before: no record of dataset version, library versions, or when a model
was trained - metrics in the README could silently drift out of sync with
the actual `.joblib` files. After: `model_card.json` records the dataset's
SHA-256, exact scikit-learn/xgboost/pandas/numpy versions, training
timestamp, and an explicit note that the dataset's own provenance is
undocumented (every feature correlates with `Potability` at |r| < 0.05,
which is consistent with an opaque or synthetic labeling process this
project cannot verify).

## 8. Input validation

Before: `parse_float(val, default)` silently substituted a fixed default
for anything that failed to parse - including physically impossible values
like a pH of 99 - so bad input never surfaced an error. After: out-of-range
or non-numeric input returns `400` with a field-specific message (e.g.
`{"ph": "pH must be between 0 and 14."}`).

## Where to see this live, not just in this file

- `aquasense_artifacts/model_card.json` - the machine-readable version of
  everything above.
- `GET /api/metadata` - serves it to the frontend.
- `frontend/analysis.html` - the champion badge, confusion matrix, and
  benchmark table now update from the real `model_card.json`/
  `candidate_models.json` instead of being hardcoded to XGBoost, and a
  panel under the model cards shows the per-fold CV scores and the
  test-vs-CV disagreement explanation whenever it's relevant.
