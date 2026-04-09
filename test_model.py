import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

# ─────────────────────────────────────────────
# PATHS  —  same as app.py
# ─────────────────────────────────────────────
MODEL_PATH   = os.path.join("model", "model.pkl")
SCALER_PATH  = os.path.join("model", "scaler.pkl")
FEATURE_COLS = ["GRE Score", "TOEFL Score", "University Rating", "SOP", "LOR", "CGPA", "Research"]

# Auto-detect correct column names from dataset
def _fix_feature_cols(df_cols):
    col_map = {c.strip().lower(): c for c in df_cols}
    fixed = []
    for f in FEATURE_COLS:
        if f in df_cols:
            fixed.append(f)
        elif f.strip().lower() in col_map:
            fixed.append(col_map[f.strip().lower()])
        else:
            print(f"⚠ Could not find column: '{f}' — columns available: {list(df_cols)}")
            fixed.append(f)
    return fixed

# ─────────────────────────────────────────────
# STEP 1 — Load model and scaler
# ─────────────────────────────────────────────
print("=" * 55)
print("  GRADPATH — MODEL ACCURACY TEST")
print("=" * 55)

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    print("❌ model.pkl or scaler.pkl not found.")
    print("   Run the app first and click 'Run Prediction' to train the model.")
    exit()

with open(MODEL_PATH,  "rb") as f: model  = pickle.load(f)
with open(SCALER_PATH, "rb") as f: scaler = pickle.load(f)
print("✅ Model and scaler loaded successfully")

# ─────────────────────────────────────────────
# STEP 2 — Load all datasets
# ─────────────────────────────────────────────
dfs = []
for fname in os.listdir("model"):
    if fname.endswith(".csv"):
        path = os.path.join("model", fname)
        try:
            tmp = pd.read_csv(path)
            tmp.columns = tmp.columns.str.strip()
            if "Chance of Admit" in tmp.columns:
                dfs.append(tmp)
                print(f"✅ Loaded dataset: {fname} ({len(tmp)} rows)")
        except Exception as e:
            print(f"⚠ Could not read {fname}: {e}")

if not dfs:
    print("❌ No valid dataset found in model/ folder.")
    exit()

df = pd.concat(dfs, ignore_index=True).drop_duplicates()
df.columns = df.columns.str.strip()
print(f"✅ Combined total: {len(df)} rows\n")
print(f"   Columns detected: {list(df.columns)}\n")

# Auto-fix column names to match what's actually in the dataset
FEATURE_COLS_ACTUAL = _fix_feature_cols(df.columns)

X = df[FEATURE_COLS_ACTUAL].astype(float)
y = df["Chance of Admit"].astype(float)

# ─────────────────────────────────────────────
# STEP 3 — Run predictions on entire dataset
# ─────────────────────────────────────────────
X_scaled     = scaler.transform(X)
y_pred       = model.predict(X_scaled)
y_pred       = np.clip(y_pred, 0, 1)

mae          = mean_absolute_error(y, y_pred)
r2           = r2_score(y, y_pred)
pct_accuracy = round((1 - mae) * 100, 2)
pct_mae      = round(mae * 100, 4)

print("=" * 55)
print("  RESULTS")
print("=" * 55)
print(f"  Total rows tested    : {len(y)}")
print(f"  MAE                  : {round(mae, 4)}")
print(f"  Average error        : ±{pct_mae}%")
print(f"  Accuracy             : {pct_accuracy}%")
print(f"  R² Score             : {round(r2, 4)}  (1.0 = perfect)")
print("=" * 55)

# ─────────────────────────────────────────────
# STEP 4 — Manual spot tests with known inputs
# ─────────────────────────────────────────────
# Format: [GRE, TOEFL, Uni Rating, SOP, LOR, CGPA, Research]
# These are real-ish profiles so you can sanity check the output

test_cases = [
    {"label": "Strong applicant",   "inputs": [330, 115, 5, 4.5, 4.5, 9.5, 1], "expected": "~0.92"},
    {"label": "Average applicant",  "inputs": [310, 100, 3, 3.0, 3.0, 8.0, 0], "expected": "~0.65"},
    {"label": "Weak applicant",     "inputs": [290,  90, 1, 1.5, 1.5, 6.5, 0], "expected": "~0.35"},
    {"label": "High GRE, low CGPA", "inputs": [325,  95, 3, 2.5, 2.5, 7.0, 0], "expected": "~0.55"},
    {"label": "Research boost",     "inputs": [305,  98, 3, 3.5, 3.5, 7.8, 1], "expected": "~0.68"},
]

print("\n  SPOT CHECKS — Manual Profile Tests")
print("=" * 55)
print(f"  {'Profile':<22} {'Predicted':>10} {'Expected':>10} {'Result':>8}")
print("-" * 55)

all_passed = True
for tc in test_cases:
    raw        = np.array([tc["inputs"]], dtype=float)
    scaled     = scaler.transform(raw)
    pred       = float(np.clip(model.predict(scaled)[0], 0, 1))
    pred_pct   = round(pred * 100, 1)
    exp        = tc["expected"]

    # Sanity check — just verify direction makes sense
    exp_val    = float(exp.replace("~",""))
    diff       = abs(pred - exp_val)
    status     = "✅ Pass" if diff < 0.15 else "⚠ Check"
    if diff >= 0.15: all_passed = False

    print(f"  {tc['label']:<22} {pred_pct:>9}%  {exp:>10}  {status:>8}")

print("=" * 55)
if all_passed:
    print("  ✅ All spot checks passed — model is behaving correctly")
else:
    print("  ⚠ Some spot checks deviated — consider retraining")
print("=" * 55)