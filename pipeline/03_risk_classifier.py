"""
PIPELINE STEP 03 - Risk Classifier (XGBoost)
=============================================
Input : data/processed/district_features.csv  (from 02_feature_engineering.py)
        OR data/processed/karnataka_clean.csv  (pre-cleaned district data)
Output: data/processed/risk_scores.csv
        data/processed/xgb_risk_model.pkl
        data/processed/xgb_results.png

Run:
    python pipeline/03_risk_classifier.py
"""

import sys
import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb

# Fix Windows console encoding (matches 01_preprocess.py)
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# --- Paths -------------------------------------------------------------------
ROOT     = Path(__file__).resolve().parent.parent
PROC_DIR = ROOT / "data" / "processed"


# --- Load ----------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    for fname in ["district_features.csv", "risk_predictions.csv", "karnataka_clean.csv"]:
        p = PROC_DIR / fname
        if p.exists():
            print(f"  [DATA] Loading {fname}")
            return pd.read_csv(p, low_memory=False)
    raise FileNotFoundError("No district feature file found. Run 02_feature_engineering.py first.")


# --- Label Creation ------------------------------------------------------------
def create_risk_labels(df: pd.DataFrame, crime_col: str = "TOTAL_CRIMES") -> pd.DataFrame:
    """Create 3-class risk label based on crime volume percentiles."""
    df = df.copy()
    if "RISK_LABEL" in df.columns:
        print("  [INFO] Using existing RISK_LABEL column")
        return df

    q33 = df[crime_col].quantile(0.33)
    q66 = df[crime_col].quantile(0.66)

    def label(v):
        if v <= q33:   return "LOW"
        elif v <= q66: return "MEDIUM"
        else:          return "HIGH"

    df["RISK_LABEL"] = df[crime_col].apply(label)
    print(f"  [LABELS] LOW: {(df['RISK_LABEL']=='LOW').sum()}  "
          f"MEDIUM: {(df['RISK_LABEL']=='MEDIUM').sum()}  "
          f"HIGH: {(df['RISK_LABEL']=='HIGH').sum()}")
    return df


# --- Feature Selection ----------------------------------------------------------
# NOTE: TOTAL_CRIMES is deliberately NOT in this list. It was previously
# included here AND used to construct RISK_LABEL above -- that's label
# leakage (the model was being handed the answer as an input), which is
# why accuracy looked artificially close to 100%. Sub-crime counts and
# derived indices are fine since they only correlate with, rather than
# directly define, the label.
FEATURE_CANDIDATES = [
    # Core crime counts
    "MURDER", "RAPE", "KIDNAPPING_ABDUCTION", "DACOITY", "ROBBERY",
    "BURGLARY", "THEFT", "AUTO_THEFT", "CHEATING", "ARSON",
    "DOWRY_DEATHS", "ASSAULT_ON_WOMEN_WITH_INTENT_TO_OUTRAGE_HER_MODESTY",
    "CRUELTY_BY_HUSBAND_OR_HIS_RELATIVES", "RIOTS",
    "CAUSING_DEATH_BY_NEGLIGENCE", "OTHER_IPC_CRIMES",
    # Derived features (aggregates of the above, not the label itself)
    "VIOLENT_CRIME_INDEX", "WOMEN_CRIME_INDEX", "PROPERTY_CRIME_INDEX",
    "CRIME_YOY_CHANGE", "CRIME_3YR_AVG",
    # Geo (optional)
    "LAT", "LON",
    # Year
    "YEAR",
]


def select_features(df: pd.DataFrame):
    available = [c for c in FEATURE_CANDIDATES if c in df.columns]
    print(f"  [FEATURES] Using {len(available)} features: {available[:8]}...")
    return df[available].copy(), available


# --- Train -----------------------------------------------------------------------
def train_xgb(df: pd.DataFrame, features: list):
    label_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}

    if df["RISK_LABEL"].dtype == object:
        y = df["RISK_LABEL"].map(label_order).fillna(0).astype(int)
    else:
        y = df["RISK_LABEL"].astype(int)

    X = df[features].fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=2.0,
        eval_metric="mlogloss",
        random_state=42,
        verbosity=0,
    )
    # use_label_encoder=False removed: deprecated/ignored in current xgboost
    # versions, which is what produced the
    # "Parameters: { 'use_label_encoder' } are not used." warning you saw.
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    train_acc = (model.predict(X_train) == y_train).mean()
    test_acc = (model.predict(X_test) == y_test).mean()
    print(f"\n  [TRAIN ACCURACY] {train_acc:.2%}")
    print(f"  [TEST ACCURACY]  {test_acc:.2%}")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["LOW", "MEDIUM", "HIGH"], zero_division=0))

    return model, X, y, X_test, y_test, y_pred, features


# --- Save Outputs -------------------------------------------------------------------
def save_model(model, path: Path):
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  [MODEL] Saved -> {path}")


def save_predictions(df: pd.DataFrame, model, X: pd.DataFrame, features: list):
    """Add predicted risk columns to original df and save."""
    label_map = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}

    probs = model.predict_proba(X.fillna(0))
    df = df.copy()
    df["PREDICTED_RISK_CODE"]  = model.predict(X.fillna(0))
    df["PREDICTED_RISK_LABEL"] = df["PREDICTED_RISK_CODE"].map(label_map)
    df["PROB_LOW"]    = probs[:, 0].round(3)
    df["PROB_MEDIUM"] = probs[:, 1].round(3)
    df["PROB_HIGH"]   = probs[:, 2].round(3)

    out = PROC_DIR / "risk_scores.csv"
    df.to_csv(out, index=False)
    print(f"  [OUT] {out}")
    return df


def save_results_chart(model, features: list, X_test, y_test, y_pred):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#161b22")

    ax = axes[0]
    ax.set_facecolor("#161b22")
    importance = pd.Series(model.feature_importances_, index=features).nlargest(12)
    colors = ["#f85149" if i < 3 else "#e3b341" if i < 6 else "#58a6ff"
              for i in range(len(importance))]
    ax.barh(importance.index[::-1], importance.values[::-1], color=colors[::-1])
    ax.set_title("Top Feature Importances", color="#e6edf3", fontsize=12)
    ax.tick_params(colors="#8b949e", labelsize=8)
    for sp in ax.spines.values(): sp.set_color("#30363d")
    ax.grid(True, alpha=0.2, color="#30363d", axis="x")

    ax2 = axes[1]
    ax2.set_facecolor("#161b22")
    cm = confusion_matrix(y_test, y_pred)
    im = ax2.imshow(cm, cmap="YlOrRd", aspect="auto")
    ax2.set_xticks([0, 1, 2]); ax2.set_yticks([0, 1, 2])
    ax2.set_xticklabels(["LOW", "MEDIUM", "HIGH"], color="#8b949e")
    ax2.set_yticklabels(["LOW", "MEDIUM", "HIGH"], color="#8b949e")
    for i in range(3):
        for j in range(3):
            ax2.text(j, i, cm[i, j], ha="center", va="center",
                     color="white", fontsize=14, fontweight="bold")
    ax2.set_title("Confusion Matrix", color="#e6edf3", fontsize=12)
    for sp in ax2.spines.values(): sp.set_color("#30363d")
    plt.colorbar(im, ax=ax2)

    plt.tight_layout()
    out = PROC_DIR / "xgb_results.png"
    plt.savefig(out, dpi=120, bbox_inches="tight", facecolor="#161b22")
    plt.close()
    print(f"  [CHART] {out}")


# --- Main ----------------------------------------------------------------------------
def main():
    print("\n" + "=" * 60)
    print("  STEP 03 - XGBoost Risk Classifier")
    print("=" * 60)

    df = load_data()
    print(f"  [DATA] Shape: {df.shape}")

    crime_col = "TOTAL_CRIMES" if "TOTAL_CRIMES" in df.columns else df.select_dtypes("number").columns[0]
    df = create_risk_labels(df, crime_col)

    df_feat, features = select_features(df)
    df_feat["RISK_LABEL"] = df["RISK_LABEL"].values

    model, X, y, X_test, y_test, y_pred, features = train_xgb(df_feat, features)

    save_model(model, PROC_DIR / "xgb_risk_model.pkl")
    save_predictions(df, model, X, features)
    save_results_chart(model, features, X_test, y_test, y_pred)

    print("\n  [DONE] Risk classifier complete.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()