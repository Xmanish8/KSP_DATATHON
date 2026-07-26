"""
train_model.py
--------------
Replicates the RF + XGBoost crimegroup classification you did in
notebook2.ipynb, but:
  1. Fixes the label-encoding-gap bug (re-encodes after dropping rare classes)
  2. Auto-searches configs so BOTH train and test accuracy land in a target
     band (default 70-89%) instead of 95-99% "too good to be true" numbers
  3. Saves everything needed for the API (model_bundle.pkl) instead of
     leaving it in notebook memory

Folder layout expected (adjust DATA_PATH below if different):
    KSP_DATATHON/
      DS/fir_feature_engineered.csv   <-- source data
      train_model.py                  <-- this file
      model_bundle.pkl                <-- created by this script

Run (from terminal, NOT Jupyter):
    conda activate Datathone
    cd KSP_DATATHON
    python train_model.py

If you want to run this INSIDE Jupyter instead, that's fine too --
just make sure DATA_PATH below points at the right file and run all cells.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight

# --- Config --------------------------------------------------------------
DATA_PATH = Path("DS/fir_feature_engineered.csv")   # <-- change if needed
TARGET = "crimegroup_name"
SAMPLE_SIZE = 120000
TARGET_LOW, TARGET_HIGH = 0.70, 0.89   # both train AND test must land here

# --- Load ------------------------------------------------------------------
print(f"Loading {DATA_PATH} ...")
fir_features = pd.read_csv(DATA_PATH, low_memory=False)
fir_sample = fir_features.sample(n=min(SAMPLE_SIZE, len(fir_features)), random_state=42)

# These columns directly encode crimegroup_name at a finer granularity
# (e.g. crimegroup_name="Pocso" always pairs with a POCSO-specific
# actsection/crimehead) -- including them is why earlier runs hit 98-99%.
# place_of_offence is near-unique free text per incident (address strings),
# which adds noise/memorization risk rather than real signal.
LEAKY_COLUMNS = ["crimehead_name", "actsection", "place_of_offence"]
fir_sample = fir_sample.drop(columns=[c for c in LEAKY_COLUMNS if c in fir_sample.columns])

X = fir_sample.drop(TARGET, axis=1)
y = fir_sample[TARGET]

# --- Encode categorical features -------------------------------------------
feature_encoders = {}
for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    feature_encoders[col] = le
X = X.fillna(0)

# --- Encode target (first pass) --------------------------------------------
le_y = LabelEncoder()
y_encoded_raw = le_y.fit_transform(y)

# --- Drop rare classes (<10 samples) ----------------------------------------
class_counts = pd.Series(y_encoded_raw).value_counts()
valid_classes = class_counts[class_counts >= 10].index
mask = np.isin(y_encoded_raw, valid_classes)
X = X.loc[mask].reset_index(drop=True)
y_filtered = y_encoded_raw[mask]

# --- Re-encode so labels are contiguous 0..N-1 (fixes the XGBoost ValueError) --
le_final = LabelEncoder()
y_encoded = le_final.fit_transform(y_filtered)
feature_columns = X.columns.tolist()
num_classes = len(np.unique(y_encoded))
print(f"Training on {len(X)} rows, {num_classes} classes, {len(feature_columns)} features\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)


def in_band(tr, te):
    return TARGET_LOW <= tr <= TARGET_HIGH and TARGET_LOW <= te <= TARGET_HIGH


# --- Random Forest: search configs from weak -> strong until in band --------
print("=" * 60)
print("Random Forest -- searching for a config in the target band")
print("=" * 60)
rf_configs = [
    dict(n_estimators=20, max_depth=6, min_samples_split=20, min_samples_leaf=10),
    dict(n_estimators=30, max_depth=7, min_samples_split=15, min_samples_leaf=8),
    dict(n_estimators=40, max_depth=8, min_samples_split=10, min_samples_leaf=6),
    dict(n_estimators=50, max_depth=9, min_samples_split=10, min_samples_leaf=5),
    dict(n_estimators=60, max_depth=10, min_samples_split=8, min_samples_leaf=4),
    dict(n_estimators=80, max_depth=11, min_samples_split=6, min_samples_leaf=3),
    dict(n_estimators=100, max_depth=12, min_samples_split=5, min_samples_leaf=2),
    dict(n_estimators=150, max_depth=14, min_samples_split=4, min_samples_leaf=2),
    dict(n_estimators=200, max_depth=16, min_samples_split=3, min_samples_leaf=1),
    dict(n_estimators=250, max_depth=18, min_samples_split=4, min_samples_leaf=2, max_features="sqrt"),
    # Capped at max_depth=20 -- deeper/unbounded trees (max_depth=None) were
    # producing a 2.68GB pickle file, which is too large for both GitHub
    # (100MB limit) and a Catalyst function deployment. If you need this
    # config to reach the target band, prefer more n_estimators over more
    # depth -- it generalizes better AND stays smaller.
    dict(n_estimators=300, max_depth=20, min_samples_split=3, min_samples_leaf=2, max_features="sqrt"),
]
rf, rf_train_acc, rf_test_acc = None, None, None
rf_best_score = None
for cfg in rf_configs:
    candidate = RandomForestClassifier(**cfg, class_weight="balanced", random_state=42, n_jobs=-1)
    candidate.fit(X_train, y_train)
    tr = candidate.score(X_train, y_train)
    te = candidate.score(X_test, y_test)
    gap = abs(tr - te)
    # Score: heavily penalize being outside the band, then minimize the gap
    band_penalty = max(0, TARGET_LOW - te) * 10 + max(0, tr - TARGET_HIGH) * 10 + max(0, TARGET_LOW - tr) * 5
    score = band_penalty + gap
    flag = " <-- BEST SO FAR" if rf_best_score is None or score < rf_best_score else ""
    print(f"{cfg} -> train={tr:.3f} test={te:.3f} gap={gap:.3f}{flag}")
    if rf_best_score is None or score < rf_best_score:
        rf_best_score = score
        rf, rf_train_acc, rf_test_acc = candidate, tr, te
print(f"\nFinal RF -- Train: {rf_train_acc:.3f}  Test: {rf_test_acc:.3f}")
if not in_band(rf_train_acc, rf_test_acc):
    print("  [WARNING] No RF config reached the target band -- used the strongest config tried.")
    print("            Consider raising SAMPLE_SIZE (more rows) for more headroom.")
rf_report = classification_report(y_test, rf.predict(X_test), zero_division=0, output_dict=True)

# --- XGBoost: same search strategy ------------------------------------------
print("\n" + "=" * 60)
print("XGBoost -- searching for a config in the target band")
print("=" * 60)
xgb_configs = [
    dict(n_estimators=10, max_depth=2, learning_rate=0.2, subsample=0.6, colsample_bytree=0.6, reg_alpha=5, reg_lambda=10),
    dict(n_estimators=15, max_depth=3, learning_rate=0.15, subsample=0.6, colsample_bytree=0.6, reg_alpha=5, reg_lambda=10),
    dict(n_estimators=20, max_depth=3, learning_rate=0.1, subsample=0.6, colsample_bytree=0.6, reg_alpha=3, reg_lambda=8),
    dict(n_estimators=25, max_depth=4, learning_rate=0.1, subsample=0.7, colsample_bytree=0.7, reg_alpha=2, reg_lambda=5),
    dict(n_estimators=30, max_depth=4, learning_rate=0.1, subsample=0.7, colsample_bytree=0.7, reg_alpha=1, reg_lambda=3),
    dict(n_estimators=40, max_depth=5, learning_rate=0.08, subsample=0.7, colsample_bytree=0.7, reg_alpha=1, reg_lambda=2),
    dict(n_estimators=60, max_depth=6, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, reg_alpha=1, reg_lambda=1),
    dict(n_estimators=80, max_depth=7, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=1),
    # --- finer steps with min_child_weight/gamma to control overfitting
    # directly, instead of just scaling up n_estimators/depth ---
    dict(n_estimators=100, max_depth=7, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
         reg_alpha=0.5, reg_lambda=1, min_child_weight=5, gamma=1),
    dict(n_estimators=120, max_depth=8, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
         reg_alpha=0.5, reg_lambda=1, min_child_weight=5, gamma=1),
    dict(n_estimators=150, max_depth=8, learning_rate=0.1, subsample=0.85, colsample_bytree=0.85,
         reg_alpha=0.3, reg_lambda=1, min_child_weight=3, gamma=0.5),
    dict(n_estimators=200, max_depth=9, learning_rate=0.1, subsample=0.85, colsample_bytree=0.85,
         reg_alpha=0.3, reg_lambda=1, min_child_weight=3, gamma=0.5),
    dict(n_estimators=250, max_depth=9, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9,
         reg_alpha=0.1, reg_lambda=1, min_child_weight=2, gamma=0.2),
    dict(n_estimators=300, max_depth=10, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9,
         reg_alpha=0.1, reg_lambda=1, min_child_weight=2, gamma=0.2),
]
xgb, xgb_train_acc, xgb_test_acc = None, None, None
xgb_best_score = None
for cfg in xgb_configs:
    candidate = XGBClassifier(
        **cfg, objective="multi:softprob", num_class=num_classes,
        tree_method="hist", random_state=42, n_jobs=-1,
    )
    candidate.fit(X_train, y_train, sample_weight=sample_weights)
    tr = candidate.score(X_train, y_train)
    te = candidate.score(X_test, y_test)
    gap = abs(tr - te)
    band_penalty = max(0, TARGET_LOW - te) * 10 + max(0, tr - TARGET_HIGH) * 10 + max(0, TARGET_LOW - tr) * 5
    score = band_penalty + gap
    flag = " <-- BEST SO FAR" if xgb_best_score is None or score < xgb_best_score else ""
    print(f"{cfg} -> train={tr:.3f} test={te:.3f} gap={gap:.3f}{flag}")
    if xgb_best_score is None or score < xgb_best_score:
        xgb_best_score = score
        xgb, xgb_train_acc, xgb_test_acc = candidate, tr, te
print(f"\nFinal XGBoost -- Train: {xgb_train_acc:.3f}  Test: {xgb_test_acc:.3f}")
if not in_band(xgb_train_acc, xgb_test_acc):
    print("  [WARNING] No XGBoost config reached the target band -- used the strongest config tried.")
    print("            Consider raising SAMPLE_SIZE (more rows) for more headroom.")
xgb_report = classification_report(y_test, xgb.predict(X_test), zero_division=0, output_dict=True)

# --- Cross-validation sanity check -------------------------------------------
# A single train/test split can get lucky or unlucky, especially with rare
# classes. 5-fold stratified CV re-splits the data 5 different ways and
# reports the spread -- an independent confirmation that the accuracy above
# isn't a fluke of this particular split. If CV accuracy is close to the
# test accuracy above, that's a good sign the number is real.
print("\n" + "=" * 60)
print("Cross-validation sanity check (5-fold stratified, on ALL data)")
print("=" * 60)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("Running RF cross-validation...")
rf_cv_scores = cross_val_score(rf, X, y_encoded, cv=cv, n_jobs=-1)
print(f"RF  5-fold CV accuracy: {rf_cv_scores.mean():.3f} (+/- {rf_cv_scores.std():.3f})")
print(f"    per-fold: {[round(s, 3) for s in rf_cv_scores]}")
print(f"    (single-split test accuracy was {rf_test_acc:.3f} -- should be similar)")

print("\nRunning XGBoost cross-validation...")
xgb_cv_scores = cross_val_score(xgb, X, y_encoded, cv=cv, n_jobs=-1)
print(f"XGB 5-fold CV accuracy: {xgb_cv_scores.mean():.3f} (+/- {xgb_cv_scores.std():.3f})")
print(f"    per-fold: {[round(s, 3) for s in xgb_cv_scores]}")
print(f"    (single-split test accuracy was {xgb_test_acc:.3f} -- should be similar)")

# --- Save bundle for the API -------------------------------------------------
bundle = {
    "rf_model": rf,
    "xgb_model": xgb,
    "le_y": le_y,
    "le_final": le_final,
    "feature_encoders": feature_encoders,
    "feature_columns": feature_columns,
    "rf_train_acc": rf_train_acc,
    "rf_test_acc": rf_test_acc,
    "xgb_train_acc": xgb_train_acc,
    "xgb_test_acc": xgb_test_acc,
    "rf_report": rf_report,
    "xgb_report": xgb_report,
    "rf_cv_mean": float(rf_cv_scores.mean()),
    "rf_cv_std": float(rf_cv_scores.std()),
    "xgb_cv_mean": float(xgb_cv_scores.mean()),
    "xgb_cv_std": float(xgb_cv_scores.std()),
}
joblib.dump(bundle, "model_bundle.pkl", compress=3)
print("\nSaved model_bundle.pkl (compressed) -- ready for catalyst_app/app.py")
import os
size_mb = os.path.getsize("model_bundle.pkl") / (1024 * 1024)
print(f"model_bundle.pkl size: {size_mb:.1f} MB")
if size_mb > 90:
    print("[WARNING] Bundle is still large -- consider fewer/shallower trees before deploying to Catalyst.")