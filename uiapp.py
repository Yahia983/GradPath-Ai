import streamlit as st
import json
import os
import datetime
import random
import hashlib
import math
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="GradPath AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS  –  dark academic / editorial look
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* App background */
.stApp {
    background: #0d0f14;
    color: #e8e4dc;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #13161e !important;
    border-right: 1px solid #2a2d38;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stButton button {
    color: #c8c4bc !important;
}

/* Headings */
h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
    letter-spacing: -0.02em;
}

/* Cards / panels */
.grad-card {
    background: #181c26;
    border: 1px solid #252834;
    border-radius: 12px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.2rem;
}

.grad-card-accent {
    background: linear-gradient(135deg, #1a1f2e 0%, #151922 100%);
    border: 1px solid #2e3650;
    border-left: 3px solid #c9a84c;
    border-radius: 12px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.2rem;
}

/* Inputs */
.stTextInput input,
.stNumberInput input,
.stSelectbox select,
.stTextArea textarea {
    background: #1e2230 !important;
    border: 1px solid #2e3248 !important;
    color: #e8e4dc !important;
    border-radius: 8px !important;
}

/* Buttons */
.stButton > button {
    background: #c9a84c !important;
    color: #0d0f14 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: #e0bf6a !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(201,168,76,0.3) !important;
}

/* Sidebar buttons – ghost style */
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    color: #a0a8c0 !important;
    border: 1px solid #2a2d38 !important;
    font-size: 0.82rem !important;
    padding: 0.35rem 0.75rem !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    border-color: #c9a84c !important;
    color: #c9a84c !important;
    background: transparent !important;
    transform: none;
    box-shadow: none !important;
}

/* Chat bubbles */
[data-testid="stChatMessage"] {
    background: #181c26 !important;
    border-radius: 12px !important;
    border: 1px solid #252834 !important;
    margin-bottom: 0.6rem !important;
}

/* Sliders */
.stSlider .st-bf { background: #c9a84c !important; }

/* Expander */
.streamlit-expanderHeader {
    background: #181c26 !important;
    border: 1px solid #252834 !important;
    border-radius: 8px !important;
    color: #e8e4dc !important;
    font-family: 'Playfair Display', serif !important;
}

/* Progress / metric */
.stMetric { background: #181c26; border-radius: 10px; padding: 0.8rem 1rem; }

/* Divider */
hr { border-color: #252834 !important; }

/* Gold accent text */
.gold { color: #c9a84c; }
.muted { color: #6b7080; font-size: 0.85rem; }

/* Score badge */
.score-badge {
    display: inline-block;
    background: linear-gradient(135deg, #c9a84c, #e0bf6a);
    color: #0d0f14;
    font-weight: 700;
    font-size: 2rem;
    padding: 0.6rem 1.4rem;
    border-radius: 50px;
    font-family: 'Playfair Display', serif;
}

/* Probability bar wrapper */
.prob-bar-wrap {
    background: #252834;
    border-radius: 6px;
    height: 10px;
    width: 100%;
    margin-top: 0.4rem;
}
.prob-bar-fill {
    height: 10px;
    border-radius: 6px;
    background: linear-gradient(90deg, #c9a84c, #e0bf6a);
    transition: width 0.6s ease;
}

/* Warning / info boxes */
.info-box {
    background: #1a1f2e;
    border-left: 3px solid #4a90d9;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    font-size: 0.88rem;
    color: #a8b8d8;
    margin: 0.6rem 0;
}
.warn-box {
    background: #1e1a10;
    border-left: 3px solid #c9a84c;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    font-size: 0.88rem;
    color: #c8b87a;
    margin: 0.6rem 0;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FILE STORAGE
# ─────────────────────────────────────────────
DATA_DIR = "data"
PROFILE_FILE = os.path.join(DATA_DIR, "profiles.json")
CHAT_FILE    = os.path.join(DATA_DIR, "chats.json")
os.makedirs(DATA_DIR, exist_ok=True)

def load_json(file):
    if not os.path.exists(file):
        return {}
    with open(file, "r") as f:
        return json.load(f)

def save_json(file, data):
    with open(file, "w") as f:
        json.dump(data, f, indent=4)

profiles = load_json(PROFILE_FILE)
chats    = load_json(CHAT_FILE)

# ─────────────────────────────────────────────
# AUTH HELPERS
# ─────────────────────────────────────────────
def hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def register(username: str, password: str) -> tuple[bool, str]:
    if not username.strip():
        return False, "Username cannot be empty."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    if username in profiles:
        return False, "Username already exists. Please log in."
    profiles[username] = {
        "password": hash_pw(password),
        "created":  str(datetime.datetime.now()),
        "academic": {}
    }
    save_json(PROFILE_FILE, profiles)
    return True, "Account created!"

def login(username: str, password: str) -> tuple[bool, str]:
    if username not in profiles:
        return False, "Username not found."
    if profiles[username]["password"] != hash_pw(password):
        return False, "Incorrect password."
    return True, "ok"

def logout():
    for key in ["user", "current_chat", "chat_id", "last_prediction"]:
        st.session_state.pop(key, None)

# ─────────────────────────────────────────────
# SESSION STATE DEFAULTS
# ─────────────────────────────────────────────
defaults = {
    "user":            None,
    "current_chat":    [],
    "chat_id":         None,
    "last_prediction": None,
    "auth_tab":        "Login",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
# PREDICTION ENGINE  —  real ML model
# ─────────────────────────────────────────────
MODEL_PATH  = os.path.join("model", "model.pkl")
SCALER_PATH = os.path.join("model", "scaler.pkl")

def _find_dataset():
    for name in ["dataset.csv", "dataset", "Admission_Predict_Ver1.1.csv", "Admission_Predict.csv"]:
        p = os.path.join("model", name)
        if os.path.exists(p):
            print(f"GRADPATH — Found dataset: {os.path.abspath(p)}")
            return p
    print(f"GRADPATH — WARNING: No dataset found in model/ folder!")
    return os.path.join("model", "dataset.csv")

DATASET_PATH = _find_dataset()
FEATURE_COLS = ["GRE Score", "TOEFL Score", "University Rating", "SOP", "LOR", "CGPA", "Research"]

@st.cache_resource(show_spinner="Training model on your dataset…")
def load_or_train_model():
    os.makedirs("model", exist_ok=True)

    print("=" * 50)
    print("GRADPATH — MODEL LOADER")
    print(f"  Model exists?  : {os.path.exists(MODEL_PATH)}")
    print(f"  Scaler exists? : {os.path.exists(SCALER_PATH)}")
    print("=" * 50)

    # ── Already trained — load from disk ──
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        print("✅ Pre-trained model found — loading from disk (skipping training)")
        with open(MODEL_PATH,  "rb") as f: model  = pickle.load(f)
        with open(SCALER_PATH, "rb") as f: scaler = pickle.load(f)
        print("✅ Model loaded successfully!")
        print("=" * 50)
        return model, scaler

    # ── Load all datasets from model/ folder ──
    dfs = []
    for fname in os.listdir("model"):
        if fname.endswith(".csv"):
            path = os.path.join("model", fname)
            try:
                tmp = pd.read_csv(path)
                tmp.columns = tmp.columns.str.strip()
                if "Chance of Admit" in tmp.columns:
                    dfs.append(tmp)
                    print(f"📂 Loaded: {fname} ({len(tmp)} rows)")
            except Exception as e:
                print(f"⚠ Could not read {fname}: {e}")

    if not dfs:
        print("❌ ERROR: No valid dataset CSV found in model/ folder!")
        print(f"   Place dataset.csv in: {os.path.abspath('model')}")
        print("=" * 50)
        return None, None

    # ── Combine all datasets ──
    df = pd.concat(dfs, ignore_index=True).drop_duplicates()
    df.columns = df.columns.str.strip()
    print(f"✅ Combined total: {len(df)} rows after deduplication")

    target = "Chance of Admit"
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"❌ ERROR: Missing columns: {missing}")
        print(f"   Columns found: {list(df.columns)}")
        print("=" * 50)
        return None, None

    X = df[FEATURE_COLS].astype(float)
    y = df[target].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"✅ Split — {len(X_train)} training rows, {len(X_test)} test rows")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    print("🧠 Training GradientBoostingRegressor — please wait...")
    model = GradientBoostingRegressor(
        n_estimators=500, learning_rate=0.03,
        max_depth=4, subsample=0.8,
        random_state=42
    )
    model.fit(X_train_s, y_train)

    mae = mean_absolute_error(y_test, model.predict(X_test_s))
    pct_mae = round(mae * 100, 2)
    print(f"✅ Training complete!")
    print(f"   MAE: {round(mae, 4)} = ~{pct_mae}% average error (lower is better)")

    with open(MODEL_PATH,  "wb") as f: pickle.dump(model,  f)
    with open(SCALER_PATH, "wb") as f: pickle.dump(scaler, f)
    print(f"✅ Model saved → {os.path.abspath(MODEL_PATH)}")
    print("=" * 50)

    return model, scaler


def predict_admission(gre: float, cgpa: float, toefl: float,
                      sop_lor: float, research: int, uni_rating: int) -> dict:

    # Always compute factor scores for bar chart and tips
    gre_score   = (gre   - 260) / 80
    cgpa_score  = cgpa / 10
    toefl_score = toefl / 120
    soplo_score = (sop_lor - 1) / 4
    res_score   = float(research)
    uni_score   = (uni_rating - 1) / 4

    model, scaler = load_or_train_model()

    if model is None:
        # ── Fallback math formula ──
        weighted = (gre_score*0.25 + cgpa_score*0.25 + toefl_score*0.15 +
                    soplo_score*0.15 + res_score*0.10 + uni_score*0.10)
        x    = (weighted - 0.6) * 10
        prob = max(0.02, min(0.98, 1/(1+math.exp(-x)) + random.uniform(-0.02, 0.02)))
    else:
        # Feature order MUST match FEATURE_COLS exactly:
        # ["GRE Score","TOEFL Score","University Rating","SOP","LOR","CGPA","Research"]
        features   = np.array([[gre, toefl, uni_rating, sop_lor, sop_lor, cgpa, research]], dtype=float)
        features_s = scaler.transform(features)
        raw  = float(model.predict(features_s)[0])
        prob = max(0.02, min(0.98, raw))

    pct = round(prob * 100, 1)

    # Band label
    if pct >= 75:
        band, colour = "High", "#5cb85c"
    elif pct >= 50:
        band, colour = "Moderate", "#c9a84c"
    elif pct >= 30:
        band, colour = "Low–Moderate", "#e8963a"
    else:
        band, colour = "Low", "#d9534f"

    # Per-factor improvement tips
    tips = []
    if gre_score   < 0.6: tips.append("GRE score — aim for 310+")
    if cgpa_score  < 0.7: tips.append("CGPA — strengthen your academic record")
    if toefl_score < 0.7: tips.append("TOEFL — target 100+ for top programs")
    if soplo_score < 0.5: tips.append("SOP / LOR quality — invest time here")
    if res_score  == 0:   tips.append("Research experience — even one project helps")

    return {
        "probability": pct,
        "band":   band,
        "colour": colour,
        "tips":   tips,
        "factors": {
            "GRE":        round(gre_score   * 100, 1),
            "CGPA":       round(cgpa_score  * 100, 1),
            "TOEFL":      round(toefl_score * 100, 1),
            "SOP / LOR":  round(soplo_score * 100, 1),
            "Research":   round(res_score   * 100, 1),
            "Uni Rating": round(uni_score   * 100, 1),
        }
    }

# ─────────────────────────────────────────────
# CHAT MANAGEMENT
# ─────────────────────────────────────────────
def save_chat():
    user = st.session_state.user
    if not user: return
    cid = st.session_state.chat_id or str(datetime.datetime.now())
    chats.setdefault(user, {})[cid] = st.session_state.current_chat
    save_json(CHAT_FILE, chats)
    st.session_state.chat_id = cid

def load_chat(cid):
    user = st.session_state.user
    st.session_state.current_chat = chats[user][cid]
    st.session_state.chat_id = cid

def new_chat():
    st.session_state.current_chat = []
    st.session_state.chat_id = None

# ─────────────────────────────────────────────
# AI RESPONSE GENERATOR
# ─────────────────────────────────────────────

# ── University database keyed by profile tier ──
UNIVERSITIES = {
    "high": {                          # pct >= 75
        "cs": {
            "reach":  ["MIT (EECS)", "Stanford (CS)", "Carnegie Mellon (MSCS)", "UC Berkeley (EECS)"],
            "match":  ["UCLA (CS)", "UC San Diego (CS)", "Georgia Tech (CS)", "University of Michigan"],
            "safe":   ["Purdue (CS)", "UT Dallas (CS)", "Northeastern (Khoury)", "ASU (CS)"],
        },
        "ds": {
            "reach":  ["Stanford (Stats/DS)", "CMU (MSML)", "Columbia (DS)", "NYU (DS)"],
            "match":  ["UC San Diego (DS)", "Georgia Tech (Analytics)", "Purdue (DS)", "UMass Amherst"],
            "safe":   ["Northeastern (DS)", "ASU (DS)", "Indiana University", "UT Arlington"],
        },
        "ee": {
            "reach":  ["MIT (EECS)", "Stanford (EE)", "Caltech (EE)", "UC Berkeley (EECS)"],
            "match":  ["Georgia Tech (EE)", "Purdue (EE)", "University of Michigan (EE)", "UCLA (EE)"],
            "safe":   ["UT Dallas (EE)", "ASU (EE)", "Northeastern (EE)", "SUNY Stony Brook"],
        },
        "general": {
            "reach":  ["MIT", "Stanford", "Carnegie Mellon", "UC Berkeley"],
            "match":  ["UCLA", "UC San Diego", "Georgia Tech", "University of Michigan"],
            "safe":   ["Purdue", "Northeastern", "ASU", "UT Dallas"],
        },
    },
    "moderate": {                      # 50 <= pct < 75
        "cs": {
            "reach":  ["Georgia Tech (CS)", "Purdue (CS)", "UT Austin (CS)", "University of Michigan"],
            "match":  ["ASU (CS)", "Northeastern (CS)", "Texas A&M (CS)", "Indiana University"],
            "safe":   ["UT Dallas (CS)", "SUNY Buffalo (CS)", "Wayne State", "UMass Lowell"],
        },
        "ds": {
            "reach":  ["Georgia Tech (Analytics)", "Purdue (DS)", "UT Austin (Stats)", "Penn State"],
            "match":  ["ASU (DS)", "Northeastern (DS)", "Indiana University (DS)", "DePaul University"],
            "safe":   ["UT Arlington (DS)", "SUNY Buffalo", "Pace University", "Wilmington University"],
        },
        "ee": {
            "reach":  ["Georgia Tech (EE)", "Purdue (EE)", "UT Austin (EE)", "Penn State (EE)"],
            "match":  ["ASU (EE)", "Texas A&M (EE)", "University of Florida (EE)", "NC State (EE)"],
            "safe":   ["UT Arlington (EE)", "SUNY Buffalo (EE)", "University of Dayton", "Wichita State"],
        },
        "general": {
            "reach":  ["Georgia Tech", "Purdue", "UT Austin", "Penn State"],
            "match":  ["ASU", "Northeastern", "Texas A&M", "Indiana University"],
            "safe":   ["UT Dallas", "SUNY Buffalo", "Wayne State", "UMass Lowell"],
        },
    },
    "low": {                           # pct < 50
        "cs": {
            "reach":  ["ASU (CS)", "Northeastern (CS)", "Stevens (CS)", "DePaul (CS)"],
            "match":  ["SUNY Buffalo (CS)", "UT Dallas (CS)", "UMass Lowell (CS)", "Pace University"],
            "safe":   ["Wayne State (CS)", "Wichita State", "South Dakota State", "Texas A&M Commerce"],
        },
        "ds": {
            "reach":  ["ASU (DS)", "Northeastern (DS)", "DePaul (DS)", "Pace University (DS)"],
            "match":  ["SUNY Buffalo", "UT Arlington (DS)", "Wilmington University", "Bellevue University"],
            "safe":   ["Regis University", "Dakota State University", "American University (DS)", "Harrisburg University"],
        },
        "ee": {
            "reach":  ["ASU (EE)", "Stevens (EE)", "University of Dayton (EE)", "Wichita State (EE)"],
            "match":  ["SUNY Buffalo (EE)", "UT Arlington (EE)", "South Dakota State", "NC A&T (EE)"],
            "safe":   ["Morgan State", "Tennessee State", "Prairie View A&M", "Alabama A&M"],
        },
        "general": {
            "reach":  ["ASU", "Northeastern", "Stevens", "DePaul"],
            "match":  ["SUNY Buffalo", "UTD", "UMass Lowell", "Pace University"],
            "safe":   ["Wayne State", "Wichita State", "South Dakota State", "Harrisburg University"],
        },
    },
}

MAJORS = {
    "cs":   ["Computer Science", "Software Engineering", "Artificial Intelligence", "Cybersecurity", "Human-Computer Interaction"],
    "ds":   ["Data Science", "Machine Learning", "Business Analytics", "Biostatistics", "Computational Social Science"],
    "ee":   ["Electrical Engineering", "Computer Engineering", "Robotics", "Signal Processing", "Power Systems"],
    "bio":  ["Bioinformatics", "Computational Biology", "Biomedical Engineering", "Neuroscience", "Genomics"],
    "fin":  ["Financial Engineering", "Quantitative Finance", "Financial Mathematics", "Risk Management", "FinTech"],
    "mgmt": ["Management of Technology", "Engineering Management", "MBA (Tech Focus)", "Supply Chain", "Operations Research"],
}

def _get_tier(pct):
    if pct >= 75:  return "high"
    if pct >= 50:  return "moderate"
    return "low"

def _detect_field(msg):
    if any(w in msg for w in ["computer science", "cs", "software", "coding", "programming", "ai", "ml", "cyber"]):
        return "cs"
    if any(w in msg for w in ["data science", "data", "analytics", "machine learning", "statistics", "stat"]):
        return "ds"
    if any(w in msg for w in ["electrical", "ee", "electronics", "robotics", "circuit", "power"]):
        return "ee"
    if any(w in msg for w in ["biology", "bio", "biomedical", "genomics", "neuro"]):
        return "bio"
    if any(w in msg for w in ["finance", "financial", "quant", "fintech", "banking"]):
        return "fin"
    if any(w in msg for w in ["management", "mba", "business", "operations", "supply chain"]):
        return "mgmt"
    return "general"

def _university_response(pct, field):
    tier  = _get_tier(pct)
    unis  = UNIVERSITIES[tier]
    group = unis.get(field, unis["general"])
    reach = ", ".join(group["reach"])
    match = ", ".join(group["match"])
    safe  = ", ".join(group["safe"])
    field_label = {"cs":"Computer Science","ds":"Data Science","ee":"Electrical Engineering",
                   "bio":"Biology/Biomedical","fin":"Finance/Quant","mgmt":"Management","general":"your field"}.get(field,"your field")
    return (
        f"Based on your **{pct}%** predicted chance, here is a balanced list for **{field_label}**:\n\n"
        f"🎯 **Reach Schools** *(apply to 3–4)*\n  • {reach.replace(', ', chr(10)+'  • ')}\n\n"
        f"✅ **Match Schools** *(apply to 4–5)*\n  • {match.replace(', ', chr(10)+'  • ')}\n\n"
        f"🛡 **Safe Schools** *(apply to 2–3)*\n  • {safe.replace(', ', chr(10)+'  • ')}\n\n"
        f"💡 Aim for **10–12 applications total** spread across all three tiers. "
        f"Would you like tips on any specific school, or help with a different field?"
    )

def _major_response(msg, academic):
    cgpa = academic.get("cgpa", 7.0)
    gre  = academic.get("gre", 300)
    res  = academic.get("research", 0)

    suggestions = []

    # Quantitative strength → CS / DS / EE
    if gre >= 315 and cgpa >= 8.0:
        suggestions += [("Computer Science / AI", "Your strong GRE quant score and CGPA make you a great fit for top CS and AI programs.")]
        suggestions += [("Data Science / ML", "High analytical scores align well with Data Science and Machine Learning programs.")]
    elif gre >= 310:
        suggestions += [("Data Science", "Your GRE score suits quantitative programs like Data Science or Analytics.")]
        suggestions += [("Electrical Engineering", "Strong quant performance translates well to EE and Computer Engineering programs.")]

    # Research experience → research-heavy fields
    if res == 1:
        suggestions += [("Bioinformatics / Computational Biology", "Research experience is highly valued — Bioinformatics and Comp Bio programs reward it strongly.")]
        suggestions += [("Human-Computer Interaction", "Research background pairs well with HCI, a growing interdisciplinary MS field.")]

    # Lower quant / broader profile → management / finance
    if cgpa >= 7.0 and gre < 310:
        suggestions += [("Engineering Management / MBA", "A solid CGPA with a broader profile fits Management of Technology or MBA (Tech Focus) well.")]
        suggestions += [("Financial Engineering", "FinTech and Financial Engineering programs value analytical skills even with a moderate GRE.")]

    if not suggestions:
        suggestions = [
            ("Computer Science", "A versatile, in-demand degree with strong job market outcomes."),
            ("Data Science", "One of the fastest growing fields — suits quantitative and analytical thinkers."),
            ("Electrical Engineering", "Broad applicability across hardware, robotics, and embedded systems."),
        ]

    lines = "\n\n".join(f"**{name}**\n  {reason}" for name, reason in suggestions[:4])
    return (
        f"Based on your academic profile, here are some majors that could be a strong fit for you:\n\n"
        f"{lines}\n\n"
        "Would you like a university list tailored to any of these fields? Just ask!"
    )

def ai_response(user_msg: str, prediction: dict | None, academic: dict) -> str:
    msg  = user_msg.lower()
    pred = prediction

    if pred:
        pct      = pred["probability"]
        band     = pred["band"]
        tips     = pred["tips"]
        tips_str = "\n".join(f"  • {t}" for t in tips) if tips else "  • Your profile looks competitive across all factors!"
        field    = _detect_field(msg)

        # ── Major suggestions ──
        if any(w in msg for w in ["major", "field", "study", "speciali", "degree", "subject", "what should i"]):
            return _major_response(msg, academic)

        # ── University / school list ──
        if any(w in msg for w in ["university", "universities", "school", "college", "program", "apply", "list", "where", "which"]):
            return _university_response(pct, field)

        # ── General improvement tips ──
        if any(w in msg for w in ["improve", "boost", "increase", "better", "chance", "tip", "advice", "suggest", "how can i"]):
            uni_hint = _university_response(pct, field)
            return (
                f"Here are your highest-leverage improvements for a **{pct}% ({band})** profile:\n\n"
                f"{tips_str}\n\n"
                f"---\n\n{uni_hint}"
            )

        # ── GRE ──
        if any(w in msg for w in ["gre", "verbal", "quant", "score"]):
            g = academic.get("gre", 300)
            if g < 305:
                return (
                    f"Your GRE of **{g}** is below average for most MS programs (median is ~310–315).\n\n"
                    "**Action plan:**\n"
                    "  • Focus on **Quantitative Reasoning** — most STEM programs weight this heavily\n"
                    "  • Use Khan Academy + Manhattan Prep for quant fundamentals\n"
                    "  • Aim for 315+ quant, 155+ verbal\n"
                    "  • A 10-point improvement can shift your predicted chance by **5–10%**"
                )
            elif g < 320:
                return (
                    f"Your GRE of **{g}** is competitive for most programs.\n\n"
                    "To push into top-tier reach schools (MIT, Stanford, CMU), aim for **320+**. "
                    "One more attempt focusing on Quant could open significantly more doors."
                )
            return f"Your GRE of **{g}** is excellent — in the top tier. Focus your energy on SOP quality and research now."

        # ── CGPA ──
        if any(w in msg for w in ["cgpa", "gpa", "grade", "academic"]):
            c = academic.get("cgpa", 7.0)
            if c < 7.0:
                return (
                    f"A CGPA of **{c}/10** is below average for competitive programs (preferred: 8.0+).\n\n"
                    "**How to compensate:**\n"
                    "  • Highlight an **upward grade trend** in your SOP\n"
                    "  • Strong GRE quant (315+) can partially offset a lower CGPA\n"
                    "  • Research experience or publications carry significant weight\n"
                    "  • Consider applying to programs that emphasize work experience over grades"
                )
            elif c < 8.0:
                return (
                    f"Your CGPA of **{c}/10** is decent but below the sweet spot for top programs (8.0+).\n\n"
                    "Highlight your strongest relevant courses and any upward trend in grades in your SOP. "
                    "A strong GRE and research experience can more than compensate."
                )
            return f"Your CGPA of **{c}/10** is strong — well above average. Make sure your SOP reflects your top coursework."

        # ── SOP / LOR ──
        if any(w in msg for w in ["sop", "statement", "lor", "recommendation", "letter", "essay"]):
            return (
                "**SOP and LOR** are often the deciding factor for borderline applicants — here is how to make them count:\n\n"
                "**Statement of Purpose (SOP):**\n"
                "  • Open with a specific research problem or moment that sparked your interest\n"
                "  • Name specific faculty at each school you want to work with — shows genuine interest\n"
                "  • Quantify every achievement ('led a team of 4', 'improved accuracy by 12%')\n"
                "  • Keep it to 1–1.5 pages — concise beats comprehensive\n"
                "  • Tailor each SOP individually — generic SOPs are easy to spot\n\n"
                "**Letters of Recommendation (LOR):**\n"
                "  • Choose professors who have seen your work directly, not just your exam scores\n"
                "  • Brief your recommenders — give them your CV, SOP draft, and key points to mention\n"
                "  • Academic letters > professional letters for research-heavy programs\n"
                "  • Submit requests at least **6–8 weeks** before deadlines"
            )

        # ── Research ──
        if any(w in msg for w in ["research", "publication", "paper", "journal", "project"]):
            r = academic.get("research", 0)
            if r == 0:
                return (
                    "You currently have **no research experience** logged — this is worth addressing.\n\n"
                    "**Quick ways to build research experience:**\n"
                    "  • Email professors at your current institution asking to assist on a project\n"
                    "  • Join a lab as a volunteer or part-time research assistant\n"
                    "  • Start a self-directed project on Kaggle or GitHub with a clear research question\n"
                    "  • Apply for summer research programs (REUs in the US, similar abroad)\n"
                    "  • Even a conference poster or workshop paper adds real weight to your application\n\n"
                    "Research experience can shift your predicted chance by **+8–15%** for top programs."
                )
            return (
                "Research experience is one of your strongest assets — here is how to leverage it:\n\n"
                "  • Describe your **specific contribution** in the SOP, not just the project title\n"
                "  • If you have a publication or preprint, mention it prominently\n"
                "  • Ask your research supervisor for a LOR — it carries more weight than a course instructor\n"
                "  • Connect your research to why you want to pursue graduate study at each specific school"
            )

        # ── TOEFL ──
        if any(w in msg for w in ["toefl", "english", "language", "ielts"]):
            t = academic.get("toefl", 90)
            if t < 90:
                return (
                    f"Your TOEFL of **{t}** is below the minimum cutoff for many programs (90+).\n\n"
                    "This could result in automatic rejection before your academic profile is even reviewed. "
                    "Retaking the TOEFL should be your **top priority** right now.\n\n"
                    "  • Focus on the **Writing** and **Speaking** sections for the fastest gains\n"
                    "  • Target 100+ for most programs, 105+ for top-tier schools"
                )
            elif t < 100:
                return (
                    f"Your TOEFL of **{t}** clears the minimum but top programs prefer 100–110+.\n\n"
                    "  • A score of 100+ removes TOEFL as a weakness in your application\n"
                    "  • Writing section improvements tend to yield the biggest score jumps\n"
                    "  • Consider one retake if you have time before application deadlines"
                )
            return f"Your TOEFL of **{t}** is strong — above the threshold for virtually all programs. No action needed here."

        # ── Scholarship / funding ──
        if any(w in msg for w in ["scholarship", "funding", "fellowship", "financial", "aid", "cost", "money"]):
            return (
                "Here are the main funding options for international graduate students:\n\n"
                "**Fellowships & Scholarships:**\n"
                "  • **Fulbright Foreign Student Program** — highly competitive, covers full tuition\n"
                "  • **NSF Graduate Research Fellowship** — for US citizens/residents\n"
                "  • University-specific merit scholarships — check each school's graduate aid page\n\n"
                "**Assistantships (most common route):**\n"
                "  • **Research Assistantship (RA)** — work with a professor, covers tuition + stipend\n"
                "  • **Teaching Assistantship (TA)** — teach undergrad sections, covers tuition + stipend\n"
                "  • Apply directly to faculty whose research matches yours — cold emailing works\n\n"
                "**Tip:** PhD programs almost always come with full funding. "
                "If cost is a major concern, consider whether a PhD path fits your goals."
            )

        # ── Timeline / deadlines ──
        if any(w in msg for w in ["deadline", "timeline", "when", "apply", "semester", "fall", "spring"]):
            return (
                "**Typical US Graduate Admissions Timeline:**\n\n"
                "  • **June–August** — Research programs and professors, shortlist 10–15 schools\n"
                "  • **August–September** — Request LORs from recommenders (give 6–8 weeks notice)\n"
                "  • **September–October** — Draft and refine SOPs, take GRE/TOEFL if needed\n"
                "  • **October–November** — Submit early decision applications\n"
                "  • **December 1–15** — Most Fall intake deadlines fall here\n"
                "  • **January–February** — Rolling decisions begin arriving\n"
                "  • **April 15** — Standard acceptance deadline (Council of Graduate Schools)\n\n"
                "**Spring intake** deadlines are typically July–August of the prior year — fewer programs offer this."
            )

    # ── No prediction yet ──
    if any(w in msg for w in ["hello", "hi", "hey", "start", "begin"]):
        return (
            "Hello! 👋 Welcome to **GradPath AI**.\n\n"
            "To get started, fill in your academic profile on the left and hit **Run Prediction**. "
            "Once you have your results, you can ask me anything — university lists, major suggestions, "
            "how to improve your GRE, SOP tips, funding options, and more."
        )

    if any(w in msg for w in ["major", "field", "study", "degree", "what should"]):
        return "Run a prediction first using your academic profile on the left, then ask me about majors — I'll tailor suggestions to your specific scores!"

    if any(w in msg for w in ["university", "school", "college", "where", "apply"]):
        return "Run a prediction first and I'll give you a full tailored university list with reach, match, and safe schools for your chosen field!"

    return (
        "I can help with:\n\n"
        "  • 🏫 **University lists** — reach, match, and safe schools by field\n"
        "  • 🎓 **Major suggestions** — based on your profile strengths\n"
        "  • 📈 **Score improvement** — GRE, TOEFL, CGPA strategies\n"
        "  • ✍️ **SOP / LOR tips** — how to write a compelling application\n"
        "  • 💰 **Funding & scholarships** — assistantships and fellowships\n"
        "  • 📅 **Application timeline** — when to do what\n\n"
        "Run a prediction first, then ask me anything!"
    )

# ── Load model on startup so terminal shows output immediately ──
_startup_model, _startup_scaler = load_or_train_model()

# ═══════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<h2 style="font-family:\'Playfair Display\',serif;color:#c9a84c;margin-bottom:0">GradPath</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color:#6b7080;font-size:0.8rem;margin-top:0">Graduate Admission Intelligence</p>', unsafe_allow_html=True)
    st.divider()

    # ── Auth ──
    if not st.session_state.user:
        tab_choice = st.radio("Account", ["Login", "Register"], horizontal=True, label_visibility="collapsed")
        st.session_state.auth_tab = tab_choice

        uname = st.text_input("Username", key="auth_uname")
        pw    = st.text_input("Password", type="password", key="auth_pw")

        if tab_choice == "Login":
            if st.button("Login →", use_container_width=True):
                ok, msg = login(uname, pw)
                if ok:
                    st.session_state.user = uname
                    st.rerun()
                else:
                    st.error(msg)
        else:
            pw2 = st.text_input("Confirm Password", type="password", key="auth_pw2")
            if st.button("Create Account →", use_container_width=True):
                if pw != pw2:
                    st.error("Passwords do not match.")
                else:
                    ok, msg = register(uname, pw)
                    if ok:
                        st.session_state.user = uname
                        st.rerun()
                    else:
                        st.error(msg)

    else:
        st.markdown(f'<p class="muted">Signed in as</p><b style="color:#e8e4dc">{st.session_state.user}</b>', unsafe_allow_html=True)

        if st.button("Sign Out", use_container_width=True):
            logout()
            st.rerun()

        st.divider()

        # ── Chat history ──
        st.markdown('<p class="muted" style="margin-bottom:0.4rem">SAVED CHATS</p>', unsafe_allow_html=True)
        user_chats = chats.get(st.session_state.user, {})
        if user_chats:
            for cid in sorted(user_chats.keys(), reverse=True)[:8]:
                label = cid[:16].replace("T", "  ").replace("-", "/")[:13]
                if st.button(f"🗂 {label}", use_container_width=True, key=f"load_{cid}"):
                    load_chat(cid)
                    st.rerun()
        else:
            st.markdown('<p class="muted">No saved chats yet.</p>', unsafe_allow_html=True)

        if st.button("＋ New Chat", use_container_width=True):
            new_chat()
            st.rerun()

# ═══════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════
if not st.session_state.user:
    # ── Landing splash ──
    st.markdown("""
    <div style="text-align:center;padding:5rem 2rem">
        <h1 style="font-family:'Playfair Display',serif;font-size:3.2rem;color:#e8e4dc;margin-bottom:0.3rem">
            GradPath <span style="color:#c9a84c">AI</span>
        </h1>
        <p style="color:#6b7080;font-size:1.1rem;max-width:520px;margin:0 auto 2rem">
            Predict your graduate school admission chances with precision. 
            Upload your profile, get your score, then ask anything.
        </p>
        <div style="display:flex;gap:2rem;justify-content:center;flex-wrap:wrap">
            <div class="grad-card" style="width:220px;text-align:left">
                <div style="font-size:1.6rem">📊</div>
                <b>Data-Driven</b>
                <p class="muted">Weighted model across 6 admission factors</p>
            </div>
            <div class="grad-card" style="width:220px;text-align:left">
                <div style="font-size:1.6rem">💬</div>
                <b>AI Chat</b>
                <p class="muted">Ask follow-up questions on your results</p>
            </div>
            <div class="grad-card" style="width:220px;text-align:left">
                <div style="font-size:1.6rem">🔒</div>
                <b>Secure Profiles</b>
                <p class="muted">Password-protected with saved chat history</p>
            </div>
        </div>
        <p style="color:#6b7080;margin-top:2rem;font-size:0.9rem">← Log in or register from the sidebar</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
# PROFILE + PREDICTION PANEL
# ─────────────────────────────────────────────
prefs = profiles[st.session_state.user].get("academic", {})

col_left, col_right = st.columns([1, 1.4], gap="large")

with col_left:
    st.markdown('<h2 style="color:#e8e4dc">Academic Profile</h2>', unsafe_allow_html=True)

    with st.form("profile_form"):
        gre = st.number_input("GRE Score (260–340)",   min_value=260, max_value=340, value=int(prefs.get("gre",  310)), step=1)
        cgpa= st.number_input("CGPA (0.0–10.0)",        min_value=0.0, max_value=10.0,value=float(prefs.get("cgpa", 7.5)), step=0.1, format="%.1f")
        toefl=st.number_input("TOEFL Score (0–120)",    min_value=0,   max_value=120, value=int(prefs.get("toefl",95)), step=1)

        st.markdown('<p style="color:#6b7080;font-size:0.82rem;margin-bottom:-0.6rem">SOP / LOR Combined Rating</p>', unsafe_allow_html=True)
        sop_lor = st.slider("", 1, 5, int(prefs.get("sop_lor", 3)), label_visibility="collapsed", key="sop_lor_slider")

        research   = st.radio("Research Experience", [0, 1], index=int(prefs.get("research", 0)),
                               format_func=lambda x: "Yes" if x else "No", horizontal=True)

        st.markdown('<p style="color:#6b7080;font-size:0.82rem;margin-bottom:-0.6rem">Target University Rating</p>', unsafe_allow_html=True)
        uni_rating = st.slider("", 1, 5, int(prefs.get("uni_rating", 3)), label_visibility="collapsed", key="uni_rating_slider")

        submitted = st.form_submit_button("Run Prediction →", use_container_width=True)

    if submitted:
        result = predict_admission(gre, cgpa, toefl, sop_lor, research, uni_rating)
        st.session_state.last_prediction = result

        # Save profile
        profiles[st.session_state.user]["academic"] = {
            "gre": gre, "cgpa": cgpa, "toefl": toefl,
            "sop_lor": sop_lor, "research": research, "uni_rating": uni_rating
        }
        save_json(PROFILE_FILE, profiles)

        # Inject prediction summary into chat
        pct   = result["probability"]
        band  = result["band"]
        tips  = result["tips"]
        tip_lines = "\n".join(f"  • {t}" for t in tips) if tips else "  • Profile looks competitive!"

        summary = (
            f"📊 **New prediction run** — {pct}% ({band} chance)\n\n"
            f"**Inputs:** GRE {gre} | CGPA {cgpa} | TOEFL {toefl} | SOP/LOR {sop_lor}/5 | "
            f"Research: {'Yes' if research else 'No'} | Uni Rating: {uni_rating}/5\n\n"
            f"**Areas to improve:**\n{tip_lines}\n\n"
            "Feel free to ask me anything about these results!"
        )

        st.session_state.current_chat.append({"role": "assistant", "content": summary})
        save_chat()
        st.rerun()

with col_right:
    st.markdown('<h2 style="color:#e8e4dc">Prediction Results</h2>', unsafe_allow_html=True)

    pred = st.session_state.last_prediction

    if not pred:
        # Try to recover from profile if previously saved
        if prefs:
            pred = predict_admission(
                prefs.get("gre",310), prefs.get("cgpa",7.5), prefs.get("toefl",95),
                prefs.get("sop_lor",3), prefs.get("research",0), prefs.get("uni_rating",3)
            )

    if pred:
        pct    = pred["probability"]
        band   = pred["band"]
        colour = pred["colour"]
        tips   = pred["tips"]

        # ── Big score ──
        st.markdown(f"""
        <div class="grad-card-accent" style="text-align:center">
            <p class="muted" style="margin-bottom:0.3rem">Predicted Acceptance Probability</p>
            <span class="score-badge">{pct}%</span>
            <p style="margin-top:0.6rem;color:{colour};font-weight:600;font-size:1rem">{band} Chance</p>
        </div>
        """, unsafe_allow_html=True)

        # ── Factor breakdown ──
        st.markdown('<div class="grad-card">', unsafe_allow_html=True)
        st.markdown('<p style="font-weight:600;margin-bottom:0.8rem">Factor Breakdown</p>', unsafe_allow_html=True)
        for factor, score in pred["factors"].items():
            bar_w = int(score)
            bar_c = "#5cb85c" if score >= 70 else "#c9a84c" if score >= 50 else "#d9534f"
            st.markdown(f"""
            <div style="margin-bottom:0.65rem">
                <div style="display:flex;justify-content:space-between;font-size:0.85rem">
                    <span>{factor}</span><span style="color:{bar_c};font-weight:600">{score}%</span>
                </div>
                <div class="prob-bar-wrap">
                    <div class="prob-bar-fill" style="width:{bar_w}%;background:{bar_c}"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Tips ──
        if tips:
            st.markdown('<div class="warn-box">⚡ <b>Priority Improvements</b><br>' +
                        "<br>".join(f"• {t}" for t in tips) + '</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-box">✅ Your profile is strong across all factors.</div>', unsafe_allow_html=True)

        st.markdown('<p class="muted">⚠ Predictions are probabilistic. Actual decisions depend on essays, interviews, and competition.</p>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="grad-card" style="text-align:center;padding:3rem">
            <div style="font-size:2rem">🎓</div>
            <p style="color:#6b7080">Fill in your academic profile on the left and hit <b>Run Prediction</b> to see your results here.</p>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CHAT SECTION
# ─────────────────────────────────────────────
st.divider()
st.markdown('<h2 style="color:#e8e4dc">Ask GradPath AI</h2>', unsafe_allow_html=True)
st.markdown('<p class="muted">Ask follow-up questions about your results — GRE strategy, SOP tips, school lists, and more.</p>', unsafe_allow_html=True)

# Display messages
for msg in st.session_state.current_chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
user_input = st.chat_input("e.g. How can I improve my chances? / Should I retake the GRE?")

if user_input:
    st.session_state.current_chat.append({"role": "user", "content": user_input})

    acad = profiles[st.session_state.user].get("academic", {})
    resp = ai_response(user_input, st.session_state.last_prediction, acad)

    st.session_state.current_chat.append({"role": "assistant", "content": resp})
    save_chat()
    st.rerun()

# ─────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────
if st.session_state.current_chat:
    st.divider()
    export_data = json.dumps({
        "user":       st.session_state.user,
        "exported":   str(datetime.datetime.now()),
        "prediction": st.session_state.last_prediction,
        "chat":       st.session_state.current_chat
    }, indent=4)
    st.download_button(
        "📥 Export Session (JSON)",
        export_data,
        file_name=f"gradpath_{st.session_state.user}_{datetime.date.today()}.json",
        mime="application/json"
    )

# ═══════════════════════════════════════════════════════════
# EXPLORE PROGRAMS, MAJORS & APPLICATION GUIDE
# ═══════════════════════════════════════════════════════════
st.divider()
st.markdown('<h2 style="color:#e8e4dc">🎓 Explore Programs, Majors & How to Apply</h2>', unsafe_allow_html=True)
st.markdown('<p class="muted">Browse by interest area — find majors you might enjoy, what careers they lead to, and exactly how to apply.</p>', unsafe_allow_html=True)

PROGRAM_DATA = {
    "💻 Computer Science & AI": {
        "description": "One of the most in-demand graduate fields. Covers algorithms, systems, AI, machine learning, and software engineering.",
        "majors": ["MS Computer Science", "MS Artificial Intelligence", "MS Machine Learning", "MS Cybersecurity", "MS Human-Computer Interaction", "MS Software Engineering"],
        "careers": ["Software Engineer", "ML Engineer", "AI Researcher", "Data Scientist", "Security Engineer", "Product Manager (Tech)"],
        "top_schools": ["MIT", "Stanford", "Carnegie Mellon", "UC Berkeley", "Georgia Tech", "University of Michigan"],
        "avg_gre": "315–330",
        "avg_cgpa": "8.5–10 / 10",
        "application_tips": [
            "Highlight coding projects on GitHub — admissions committees look at your portfolio",
            "Research the faculty you want to work with and name them specifically in your SOP",
            "LeetCode / competitive programming experience is a plus for top programs",
            "A research paper or publication significantly boosts your profile",
            "Apply to 10–14 programs spread across reach, match, and safe tiers",
        ],
        "timeline": "Deadlines: Dec 1–Jan 15 for Fall intake. Results: Feb–April.",
    },
    "📊 Data Science & Analytics": {
        "description": "Combines statistics, programming, and domain knowledge to extract insights from data. Highly interdisciplinary.",
        "majors": ["MS Data Science", "MS Business Analytics", "MS Statistics", "MS Computational Social Science", "MS Biostatistics", "MS Financial Analytics"],
        "careers": ["Data Scientist", "Data Analyst", "Business Intelligence Analyst", "Quantitative Analyst", "Research Scientist", "AI Product Manager"],
        "top_schools": ["Columbia", "NYU", "UC San Diego", "Georgia Tech", "Purdue", "Carnegie Mellon"],
        "avg_gre": "310–325",
        "avg_cgpa": "8.0–9.5 / 10",
        "application_tips": [
            "Show a Kaggle portfolio, personal projects, or GitHub repos with data analysis work",
            "Statistics and linear algebra coursework should be highlighted in your SOP",
            "Industry experience with data tools (SQL, Python, R) is a strong differentiator",
            "Some programs require a writing sample or research statement — prepare one early",
            "Target programs that match your focus: some are more business-oriented, others more technical",
        ],
        "timeline": "Deadlines: Nov 15–Jan 1 for Fall intake. Results: Feb–March.",
    },
    "⚡ Electrical & Computer Engineering": {
        "description": "Covers hardware, circuits, signal processing, embedded systems, robotics, and computer architecture.",
        "majors": ["MS Electrical Engineering", "MS Computer Engineering", "MS Robotics", "MS Signal Processing", "MS Power Systems", "MS VLSI Design"],
        "careers": ["Hardware Engineer", "Robotics Engineer", "Embedded Systems Engineer", "RF Engineer", "Power Systems Engineer", "Chip Designer"],
        "top_schools": ["MIT", "Stanford", "Caltech", "Georgia Tech", "Purdue", "University of Michigan"],
        "avg_gre": "315–330",
        "avg_cgpa": "8.0–9.5 / 10",
        "application_tips": [
            "Lab experience and hardware projects are very highly valued — mention specifics",
            "Research assistantships are common in EE — email professors directly before applying",
            "GRE Quantitative score is weighted heavily — aim for 165+",
            "Internships at hardware companies (Intel, Qualcomm, NVIDIA) strengthen your profile significantly",
            "Include any patents, publications, or conference presentations in your application",
        ],
        "timeline": "Deadlines: Dec 1–Jan 15 for Fall intake. Results: Feb–April.",
    },
    "🧬 Biomedical & Life Sciences": {
        "description": "Applies engineering and computational methods to biology, medicine, and healthcare. Fast-growing field with high impact.",
        "majors": ["MS Biomedical Engineering", "MS Bioinformatics", "MS Computational Biology", "MS Neuroscience", "MS Genomics", "MS Health Informatics"],
        "careers": ["Biomedical Engineer", "Bioinformatics Scientist", "Clinical Data Analyst", "Research Scientist", "Pharmaceutical Analyst", "Healthcare Data Engineer"],
        "top_schools": ["Johns Hopkins", "UCSF", "MIT", "Stanford", "Georgia Tech", "University of Michigan"],
        "avg_gre": "308–322",
        "avg_cgpa": "8.0–9.5 / 10",
        "application_tips": [
            "Wet lab or computational biology research experience is almost mandatory for top programs",
            "A clear research interest stated in your SOP matters more than in other fields",
            "Strong letters from research supervisors carry more weight than course instructors",
            "Some programs require GRE Biology Subject Test — check each school's requirements",
            "Highlight any exposure to bioinformatics tools (Python, R, BLAST, genome pipelines)",
        ],
        "timeline": "Deadlines: Nov 1–Dec 15 for Fall intake. Results: Jan–March.",
    },
    "💰 Financial Engineering & Quantitative Finance": {
        "description": "Applies mathematics and programming to financial markets, risk management, and investment strategies.",
        "majors": ["MS Financial Engineering", "MS Quantitative Finance", "MS Mathematical Finance", "MS Risk Management", "MS FinTech", "MS Computational Finance"],
        "careers": ["Quantitative Analyst", "Risk Manager", "Derivatives Trader", "Portfolio Manager", "FinTech Developer", "Actuarial Analyst"],
        "top_schools": ["Columbia", "NYU Courant", "Carnegie Mellon", "Princeton", "UC Berkeley", "Baruch College"],
        "avg_gre": "318–330",
        "avg_cgpa": "8.5–10 / 10",
        "application_tips": [
            "Strong mathematics background (calculus, probability, linear algebra) is essential — highlight it",
            "Programming skills in Python, C++, or MATLAB are expected — show projects",
            "Finance internships or CFA Level 1 significantly strengthen non-finance undergrad applicants",
            "Some programs require the GMAT instead of or in addition to the GRE — check each school",
            "Interview preparation is required for top MFE programs — practice stochastic calculus questions",
        ],
        "timeline": "Deadlines: Nov 1–Jan 1 for Fall intake. Results: Jan–March.",
    },
    "🏗 Engineering Management & MBA": {
        "description": "Bridges technical expertise and business leadership. Ideal for engineers who want to move into management or entrepreneurship.",
        "majors": ["MS Engineering Management", "MBA (Tech Focus)", "MS Management of Technology", "MS Operations Research", "MS Supply Chain Management", "MS Project Management"],
        "careers": ["Product Manager", "Engineering Manager", "Operations Manager", "Management Consultant", "Entrepreneur", "Supply Chain Director"],
        "top_schools": ["MIT Sloan", "Stanford GSB", "Carnegie Mellon Tepper", "Northwestern Kellogg", "Duke Fuqua", "Cornell Tech"],
        "avg_gre": "308–325",
        "avg_cgpa": "7.5–9.0 / 10",
        "application_tips": [
            "Work experience matters more here than in pure technical programs — highlight leadership roles",
            "GMAT is often preferred over GRE for MBA programs — check each school's preference",
            "Essays are more important in business programs — spend significant time on them",
            "Show clear career goals — admissions committees want to see a concrete vision",
            "Recommendation letters from managers or industry supervisors outweigh academic ones here",
        ],
        "timeline": "Round 1: Sept–Oct. Round 2 (most competitive): Jan. Round 3: March–April.",
    },
}

# ── Tabs for each field ──
tab_labels = list(PROGRAM_DATA.keys())
tabs = st.tabs(tab_labels)

for tab, label in zip(tabs, tab_labels):
    prog = PROGRAM_DATA[label]
    with tab:
        col_a, col_b = st.columns([1.1, 1], gap="large")

        with col_a:
            # Description
            st.markdown(f'<div class="grad-card"><p style="color:#a8b8d8;font-size:0.95rem">{prog["description"]}</p></div>', unsafe_allow_html=True)

            # Majors
            st.markdown('<p style="font-weight:600;color:#c9a84c;margin-bottom:0.4rem">Available Majors</p>', unsafe_allow_html=True)
            majors_html = "".join(
                f'<span style="display:inline-block;background:#1e2230;border:1px solid #2e3248;'
                f'border-radius:20px;padding:0.25rem 0.75rem;margin:0.2rem;font-size:0.82rem;color:#e8e4dc">{m}</span>'
                for m in prog["majors"]
            )
            st.markdown(f'<div style="margin-bottom:1rem">{majors_html}</div>', unsafe_allow_html=True)

            # Careers
            st.markdown('<p style="font-weight:600;color:#c9a84c;margin-bottom:0.4rem">Career Paths</p>', unsafe_allow_html=True)
            careers_html = "".join(
                f'<span style="display:inline-block;background:#1a2218;border:1px solid #2e4830;'
                f'border-radius:20px;padding:0.25rem 0.75rem;margin:0.2rem;font-size:0.82rem;color:#7ecb8f">{c}</span>'
                for c in prog["careers"]
            )
            st.markdown(f'<div style="margin-bottom:1rem">{careers_html}</div>', unsafe_allow_html=True)

            # Profile requirements
            st.markdown(f"""
            <div class="grad-card" style="margin-top:0.5rem">
                <p style="font-weight:600;color:#c9a84c;margin-bottom:0.5rem">Typical Profile Requirements</p>
                <p style="font-size:0.88rem;color:#a0a8c0;margin:0.2rem 0">📝 GRE Score: <b style="color:#e8e4dc">{prog["avg_gre"]}</b></p>
                <p style="font-size:0.88rem;color:#a0a8c0;margin:0.2rem 0">🎓 CGPA: <b style="color:#e8e4dc">{prog["avg_cgpa"]}</b></p>
                <p style="font-size:0.88rem;color:#a0a8c0;margin:0.2rem 0">📅 Timeline: <b style="color:#e8e4dc">{prog["timeline"]}</b></p>
            </div>
            """, unsafe_allow_html=True)

        with col_b:
            # Top schools
            st.markdown('<p style="font-weight:600;color:#c9a84c;margin-bottom:0.4rem">Top Schools</p>', unsafe_allow_html=True)
            for school in prog["top_schools"]:
                st.markdown(f'<p style="font-size:0.88rem;color:#e8e4dc;margin:0.15rem 0">🏛 {school}</p>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Application tips
            st.markdown('<p style="font-weight:600;color:#c9a84c;margin-bottom:0.4rem">How to Apply — Key Tips</p>', unsafe_allow_html=True)
            st.markdown('<div class="grad-card-accent">', unsafe_allow_html=True)
            for i, tip in enumerate(prog["application_tips"], 1):
                st.markdown(f'<p style="font-size:0.85rem;color:#c8c4bc;margin:0.4rem 0"><b style="color:#c9a84c">{i}.</b> {tip}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:2rem 0 1rem;color:#3a3e50;font-size:0.78rem">
    GradPath AI · Predictions are probabilistic, not deterministic · Not affiliated with any university
</div>
""", unsafe_allow_html=True)