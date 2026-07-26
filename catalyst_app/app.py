"""
app.py -- Catalyst Advanced I/O Function
=========================================
Serves the trained RF/XGBoost crime-group model as a JSON API instead of a
Streamlit page. Deploy this whole `catalyst_app/` folder as a Zoho Catalyst
Advanced I/O Function (Python).

Local test (before deploying):
    pip install -r requirements.txt
    python app.py
    # then in another terminal:
    curl http://localhost:9000/health
    curl -X POST http://localhost:9000/predict -H "Content-Type: application/json" -d "{\"model\":\"xgb\",\"features\":{...}}"

Deploy to Catalyst:
    catalyst deploy
(after `catalyst init` in this folder and setting up catalyst-config.json --
see the template included alongside this file)
"""

import os
import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDLE_PATH = os.path.join(BASE_DIR, "model_bundle.pkl")

print(f"Loading model bundle from {BUNDLE_PATH} ...")
bundle = joblib.load(BUNDLE_PATH)

rf = bundle["rf_model"]
xgb_model = bundle["xgb_model"]
le_y = bundle["le_y"]
le_final = bundle["le_final"]
feature_encoders = bundle["feature_encoders"]
feature_columns = bundle["feature_columns"]


def decode_label(encoded_id: int) -> str:
    return le_y.inverse_transform(le_final.inverse_transform([encoded_id]))[0]


def build_feature_row(payload: dict):
    """Turn a JSON dict of {feature_name: value} into a model-ready row,
    encoding categorical fields with the saved encoders."""
    row = []
    for col in feature_columns:
        val = payload.get(col, 0)
        if col in feature_encoders:
            enc = feature_encoders[col]
            val = str(val)
            if val not in enc.classes_:
                # unseen category -> fall back to the most common training value
                val = enc.classes_[0]
            val = enc.transform([val])[0]
        row.append(val)
    return np.array(row).reshape(1, -1)


@app.route("/health", methods=["GET"])
def health():
    rf_gap = abs(bundle["rf_train_acc"] - bundle["rf_test_acc"])
    xgb_gap = abs(bundle["xgb_train_acc"] - bundle["xgb_test_acc"])
    response = {
        "status": "ok",
        "rf_train_accuracy": round(bundle["rf_train_acc"], 4),
        "rf_test_accuracy": round(bundle["rf_test_acc"], 4),
        "rf_train_test_gap": round(rf_gap, 4),
        "xgb_train_accuracy": round(bundle["xgb_train_acc"], 4),
        "xgb_test_accuracy": round(bundle["xgb_test_acc"], 4),
        "xgb_train_test_gap": round(xgb_gap, 4),
        "num_features": len(feature_columns),
    }
    # CV scores are only present in bundles created by the updated train_model.py
    if "rf_cv_mean" in bundle:
        response["rf_cross_val_accuracy"] = f"{bundle['rf_cv_mean']:.4f} (+/- {bundle['rf_cv_std']:.4f})"
        response["xgb_cross_val_accuracy"] = f"{bundle['xgb_cv_mean']:.4f} (+/- {bundle['xgb_cv_std']:.4f})"
    return jsonify(response)


@app.route("/features", methods=["GET"])
def features():
    """Lets the frontend discover what fields to send, and valid categorical values.
    Options are capped at 200 per field -- some columns (village/beat names,
    officer IDs) have thousands of unique values, which made this response
    1.7MB+. Use /form-fields for a small curated set for a live demo."""
    info = {}
    for col in feature_columns:
        if col in feature_encoders:
            opts = list(feature_encoders[col].classes_)
            info[col] = {"type": "categorical", "options": opts[:200], "total_options": len(opts)}
        else:
            info[col] = {"type": "numeric"}
    return jsonify(info)


# A small, human-meaningful subset of fields for the demo page -- the rest
# of the 47 columns get sent as 0/default automatically by build_feature_row.
DEMO_FIELDS = [
    "district_name", "fir_year", "fir_month", "fir_type", "complaint_mode",
    "victim_count", "accused_count", "male", "female", "boy", "girl", "is_heinous",
]


@app.route("/form-fields", methods=["GET"])
def form_fields():
    info = {}
    for col in DEMO_FIELDS:
        if col not in feature_columns:
            continue
        if col in feature_encoders:
            opts = list(feature_encoders[col].classes_)
            info[col] = {"type": "categorical", "options": opts[:100]}
        else:
            info[col] = {"type": "numeric"}
    return jsonify(info)


DEMO_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KSP -- Crime Group Classifier</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Zilla+Slab:wght@500;700&family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --ink-navy: #0B1220;
    --panel-navy: #121B2E;
    --panel-navy-2: #17233B;
    --case-paper: #EDE6D6;
    --case-paper-ink: #2A2620;
    --siren-amber: #F2A93B;
    --evidence-red: #C1442D;
    --steel: #4B5563;
    --steel-light: #7A8699;
    --text-light: #E6EDF3;
    --text-muted: #8B96A8;
    --signal-green: #4C9A6A;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--ink-navy);
    background-image:
      linear-gradient(180deg, rgba(242,169,59,0.04), transparent 200px),
      repeating-linear-gradient(0deg, rgba(255,255,255,0.015) 0px, transparent 1px, transparent 3px);
    color: var(--text-light);
    font-family: 'Inter', sans-serif;
    min-height: 100vh;
  }
  a:focus-visible, button:focus-visible, select:focus-visible, input:focus-visible {
    outline: 2px solid var(--siren-amber);
    outline-offset: 2px;
  }
  .wrap { max-width: 880px; margin: 0 auto; padding: 28px 20px 60px; }

  /* --- Header: case file tab --- */
  .eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--siren-amber);
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .eyebrow svg { flex-shrink: 0; }
  h1 {
    font-family: 'Zilla Slab', serif;
    font-weight: 700;
    font-size: clamp(28px, 5vw, 40px);
    margin: 6px 0 4px;
    letter-spacing: -0.01em;
  }
  .case-meta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: var(--text-muted);
    display: flex;
    gap: 18px;
    flex-wrap: wrap;
    margin-bottom: 22px;
    border-bottom: 1px dashed var(--steel);
    padding-bottom: 16px;
  }
  .case-meta b { color: var(--text-light); font-weight: 500; }

  /* --- Layout --- */
  .grid { display: grid; grid-template-columns: 1fr; gap: 20px; }
  @media (min-width: 760px) {
    .grid { grid-template-columns: 1.1fr 0.9fr; align-items: start; }
  }

  .panel {
    background: var(--panel-navy);
    border: 1px solid #223049;
    border-radius: 4px;
    padding: 22px;
    position: relative;
  }
  .panel-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--steel-light);
    margin: 0 0 16px;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .panel-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #223049;
  }

  /* --- Form fields styled like FIR intake paper --- */
  .field { margin-bottom: 14px; }
  .field label {
    display: block;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-muted);
    margin-bottom: 5px;
  }
  .field select, .field input {
    width: 100%;
    padding: 10px 12px;
    background: var(--case-paper);
    color: var(--case-paper-ink);
    border: 1px solid #cfc6ac;
    border-radius: 3px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 14px;
  }
  .field select:focus, .field input:focus { border-color: var(--siren-amber); }
  .fields-two { display: grid; grid-template-columns: 1fr 1fr; gap: 0 14px; }

  /* --- Model toggle --- */
  .model-toggle {
    display: flex;
    border: 1px solid var(--steel);
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 18px;
  }
  .model-toggle button {
    flex: 1;
    padding: 10px;
    background: transparent;
    color: var(--text-muted);
    border: none;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    letter-spacing: 0.05em;
    cursor: pointer;
    border-right: 1px solid var(--steel);
  }
  .model-toggle button:last-child { border-right: none; }
  .model-toggle button.active {
    background: var(--siren-amber);
    color: #1a1204;
    font-weight: 600;
  }

  .stamp-btn {
    width: 100%;
    padding: 14px;
    margin-top: 6px;
    background: var(--evidence-red);
    color: #fff0eb;
    border: none;
    border-radius: 3px;
    font-family: 'Zilla Slab', serif;
    font-weight: 700;
    font-size: 15px;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    cursor: pointer;
    transition: transform 0.05s ease, background 0.15s ease;
  }
  .stamp-btn:hover { background: #d9502f; }
  .stamp-btn:active { transform: scale(0.98); }

  /* --- Result panel --- */
  #result-panel { display: none; }
  .scan-line {
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--siren-amber), transparent);
    animation: sweep 0.9s ease-out;
    margin-bottom: 14px;
  }
  @keyframes sweep {
    from { transform: scaleX(0); opacity: 0.2; }
    to   { transform: scaleX(1); opacity: 1; }
  }
  @media (prefers-reduced-motion: reduce) {
    .scan-line { animation: none; }
    .stamp-seal { animation: none !important; }
  }

  .stamp-seal {
    display: inline-block;
    font-family: 'Zilla Slab', serif;
    font-weight: 700;
    font-size: 20px;
    color: var(--evidence-red);
    border: 3px solid var(--evidence-red);
    padding: 8px 16px;
    border-radius: 4px;
    transform: rotate(-3deg);
    letter-spacing: 0.02em;
    text-transform: uppercase;
    animation: stamp-in 0.25s ease-out;
  }
  @keyframes stamp-in {
    from { opacity: 0; transform: rotate(-3deg) scale(1.4); }
    to   { opacity: 1; transform: rotate(-3deg) scale(1); }
  }
  .stamp-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: var(--text-muted);
    margin-top: 10px;
  }

  .ballistics { margin-top: 20px; }
  .b-row { margin-bottom: 12px; }
  .b-label-row {
    display: flex;
    justify-content: space-between;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    margin-bottom: 4px;
  }
  .b-label-row .name { color: var(--text-light); }
  .b-label-row .pct { color: var(--siren-amber); }
  .b-track {
    height: 6px;
    background: #0d1524;
    border-radius: 0;
    position: relative;
    border-bottom: 1px dashed #2a3a56;
  }
  .b-fill {
    height: 100%;
    background: var(--siren-amber);
    transition: width 0.4s ease;
  }

  .placeholder-note {
    color: var(--text-muted);
    font-size: 13px;
    font-style: italic;
  }

  /* --- Footer / accuracy strip --- */
  .footer {
    margin-top: 26px;
    padding-top: 14px;
    border-top: 1px dashed var(--steel);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: var(--text-muted);
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
  }
  .footer b { color: var(--signal-green); }
</style>
</head>
<body>
<div class="wrap">

  <div class="eyebrow">
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M12 2 L20 6 V12 C20 17 16.5 20.5 12 22 C7.5 20.5 4 17 4 12 V6 Z"/>
    </svg>
    Karnataka State Police &mdash; Predictive Unit
  </div>
  <h1>Crime Group Classifier</h1>
  <div class="case-meta" id="caseMeta">
    <span>CASE REF: <b id="caseRef">--</b></span>
    <span>MODEL: <b id="modelMetaLabel">XGBoost</b></span>
    <span>STATUS: <b style="color:var(--siren-amber);">AWAITING INTAKE</b></span>
  </div>

  <div class="grid">
    <!-- Intake form -->
    <div class="panel">
      <p class="panel-title">FIR Intake</p>

      <div class="model-toggle">
        <button id="btnXgb" class="active" onclick="selectModel('xgb')">XGBOOST</button>
        <button id="btnRf" onclick="selectModel('rf')">RANDOM FOREST</button>
      </div>

      <div id="fields"></div>
      <button class="stamp-btn" onclick="runPredict()">Classify Case</button>
    </div>

    <!-- Result -->
    <div class="panel">
      <p class="panel-title">Classification Result</p>
      <div id="placeholder" class="placeholder-note">Submit an intake to generate a classification.</div>
      <div id="result-panel">
        <div class="scan-line"></div>
        <div class="stamp-seal" id="stampSeal">--</div>
        <div class="stamp-sub" id="stampSub"></div>
        <div class="ballistics" id="ballistics"></div>
      </div>
    </div>
  </div>

  <div class="footer" id="footerStats">Loading model metadata...</div>
</div>

<script>
let currentModel = 'xgb';

function selectModel(m) {
  currentModel = m;
  document.getElementById('btnXgb').classList.toggle('active', m === 'xgb');
  document.getElementById('btnRf').classList.toggle('active', m === 'rf');
  document.getElementById('modelMetaLabel').textContent = m === 'xgb' ? 'XGBoost' : 'Random Forest';
}

function genCaseRef() {
  const y = new Date().getFullYear();
  const n = Math.floor(Math.random() * 90000 + 10000);
  return `KSP/${y}/${n}`;
}
document.getElementById('caseRef').textContent = genCaseRef();

async function loadFields() {
  const res = await fetch('/form-fields');
  const data = await res.json();
  const container = document.getElementById('fields');
  const entries = Object.entries(data);

  entries.forEach(([name, info], i) => {
    const wrapper = document.createElement('div');
    wrapper.className = 'field';
    const label = document.createElement('label');
    label.textContent = name.replace(/_/g, ' ');
    wrapper.appendChild(label);

    let input;
    if (info.type === 'categorical') {
      input = document.createElement('select');
      input.id = 'f_' + name;
      info.options.forEach(opt => {
        const o = document.createElement('option');
        o.value = opt; o.textContent = opt;
        input.appendChild(o);
      });
    } else {
      input = document.createElement('input');
      input.type = 'number';
      input.id = 'f_' + name;
      input.value = 0;
    }
    wrapper.appendChild(input);
    container.appendChild(wrapper);
  });
}

async function runPredict() {
  const featureNames = [...document.querySelectorAll('[id^=f_]')].map(el => el.id.slice(2));
  const features = {};
  featureNames.forEach(name => {
    const el = document.getElementById('f_' + name);
    features[name] = el.tagName === 'SELECT' ? el.value : Number(el.value);
  });

  document.getElementById('caseMeta').querySelector('span:last-child b').textContent = 'PROCESSING';

  const res = await fetch('/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({model: currentModel, features}),
  });
  const data = await res.json();

  document.getElementById('placeholder').style.display = 'none';
  const panel = document.getElementById('result-panel');
  panel.style.display = 'none';
  void panel.offsetWidth; // restart animation
  panel.style.display = 'block';

  document.getElementById('stampSeal').textContent = 'Classified: ' + data.predicted_crime_group;
  document.getElementById('stampSub').textContent =
    `Case ${document.getElementById('caseRef').textContent} -- model: ${data.model_used.toUpperCase()}`;
  document.getElementById('caseMeta').querySelector('span:last-child b').innerHTML =
    '<span style="color:var(--signal-green);">CLASSIFIED</span>';

  const ballistics = document.getElementById('ballistics');
  ballistics.innerHTML = '';
  if (data.top_5) {
    const maxProb = Math.max(...data.top_5.map(r => r.probability));
    data.top_5.forEach(r => {
      const pct = (r.probability / maxProb) * 100;
      const row = document.createElement('div');
      row.className = 'b-row';
      row.innerHTML = `
        <div class="b-label-row"><span class="name">${r.crime_group}</span><span class="pct">${(r.probability*100).toFixed(1)}%</span></div>
        <div class="b-track"><div class="b-fill" style="width:${pct}%"></div></div>`;
      ballistics.appendChild(row);
    });
  }
}

async function loadFooter() {
  const res = await fetch('/health');
  const d = await res.json();
  const cv = d.xgb_cross_val_accuracy ? ` | XGB 5-fold CV: <b>${d.xgb_cross_val_accuracy}</b>` : '';
  document.getElementById('footerStats').innerHTML =
    `RF test acc: <b>${(d.rf_test_accuracy*100).toFixed(1)}%</b>` +
    ` | XGB test acc: <b>${(d.xgb_test_accuracy*100).toFixed(1)}%</b>` +
    cv +
    ` | Features: <b>${d.num_features}</b>`;
}

loadFields();
loadFooter();
</script>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def demo_page():
    return DEMO_PAGE


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    if not payload or "features" not in payload:
        return jsonify({"error": "Send JSON like {'model': 'xgb', 'features': {...}}"}), 400

    model_name = payload.get("model", "xgb").lower()
    model = xgb_model if model_name == "xgb" else rf

    row = build_feature_row(payload["features"])
    pred_encoded = model.predict(row)[0]
    pred_label = decode_label(int(pred_encoded))

    response = {"model_used": model_name, "predicted_crime_group": pred_label}

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(row)[0]
        top5_idx = np.argsort(proba)[-5:][::-1]
        response["top_5"] = [
            {"crime_group": decode_label(int(i)), "probability": round(float(proba[i]), 4)}
            for i in top5_idx
        ]

    return jsonify(response)


if __name__ == "__main__":
    port = int(os.environ.get("X_ZOHO_CATALYST_LISTEN_PORT", 9000))
    app.run(host="0.0.0.0", port=port)