import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from src.config import MODELS_DIR, RESULTS_DIR, PLOTS_DIR
from src.data import load_dataset_split, load_raw_dataset

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
LOGO_PATH  = ASSETS_DIR / "lol_logo.png"

# ── CSS ─────────────────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700;900&family=Inter:wght@400;600;700&display=swap');

.stApp {
    background-color: #010a13;
    background-image: radial-gradient(ellipse at 50% 0%, rgba(200,170,110,0.07) 0%, transparent 55%);
    font-family: 'Inter', sans-serif;
    color: #a8b2bf;
}
h1,h2,h3 { font-family:'Cinzel',serif !important; color:#c8aa6e !important; letter-spacing:2px; }
.gold-line { border:none; height:2px; background:linear-gradient(90deg,transparent,#c8aa6e 20%,#f0e6d2 50%,#c8aa6e 80%,transparent); margin:28px 0; }
.thin-line  { border:none; height:1px; background:linear-gradient(90deg,transparent,rgba(200,170,110,0.3),transparent); margin:24px 0; }

/* ── Section label ── */
.sec-label {
    font-family:'Cinzel',serif; font-size:11px; letter-spacing:3px; color:#785a28;
    text-transform:uppercase; text-align:center; margin:20px 0 14px;
}

/* ── Two-team scoreboard ── */
.sb-wrap {
    background:linear-gradient(160deg,#0a1428,#091020);
    border:1px solid rgba(200,170,110,0.3); border-radius:4px;
    overflow:hidden; box-shadow:0 8px 40px rgba(0,0,0,0.8);
    margin-bottom:20px; position:relative;
}
.sb-wrap::before { content:''; position:absolute; top:0;left:0;right:0; height:2px;
    background:linear-gradient(90deg,transparent,#c8aa6e,transparent); }

/* totals header */
.sb-totals {
    display:flex; align-items:center; justify-content:space-between;
    padding:10px 24px; background:rgba(0,0,0,0.4);
    border-bottom:1px solid rgba(200,170,110,0.15);
}
.sb-totals-item { text-align:center; }
.sb-tot-label { font-size:9px; letter-spacing:2px; color:#5b5a56; text-transform:uppercase; }
.sb-tot-val   { font-size:18px; font-weight:700; }
.sb-tot-blue  { color:#4fc3f7; }
.sb-tot-red   { color:#e84057; }
.sb-tot-mid   { font-size:10px; letter-spacing:2px; color:#785a28; text-transform:uppercase; text-align:center; }

/* team section */
.sb-team-hdr {
    display:flex; align-items:center; padding:8px 20px;
    font-size:11px; font-weight:700; letter-spacing:3px; text-transform:uppercase; font-family:'Cinzel',serif;
}
.sb-team-hdr.blue { background:linear-gradient(90deg,rgba(0,68,170,0.35),transparent); color:#4fc3f7; border-bottom:1px solid rgba(79,195,247,0.2); }
.sb-team-hdr.red  { background:linear-gradient(90deg,rgba(200,30,30,0.35),transparent); color:#e84057; border-bottom:1px solid rgba(232,64,87,0.2); border-top:1px solid rgba(200,170,110,0.1); }

/* stat grid */
.sb-grid {
    display:grid; grid-template-columns:1fr 1fr 1fr 1fr 1fr 1fr;
    padding:14px 20px; gap:8px; border-bottom:1px solid rgba(255,255,255,0.04);
}
.sb-cell { text-align:center; }
.sb-cell-label { font-size:9px; letter-spacing:1.5px; color:#5b5a56; text-transform:uppercase; margin-bottom:4px; }
.sb-cell-val   { font-size:20px; font-weight:700; }
.c-blue { color:#4fc3f7; }
.c-gold { color:#c8aa6e; }
.c-teal { color:#0ac8b9; }
.c-red  { color:#e84057; }
.c-grn  { color:#4caf50; }
.c-wht  { color:#f0e6d2; }

/* diff row */
.sb-diffs {
    display:flex; justify-content:center; gap:32px;
    padding:10px 20px; background:rgba(0,0,0,0.2);
}
.diff-chip { text-align:center; }
.diff-lbl { font-size:9px; letter-spacing:1.5px; color:#5b5a56; text-transform:uppercase; }
.diff-val { font-size:15px; font-weight:700; }

/* objectives */
.sb-objs {
    display:flex; justify-content:center; gap:28px; padding:12px 20px;
    background:rgba(0,0,0,0.3); border-top:1px solid rgba(200,170,110,0.1);
}
.obj-b { display:flex; flex-direction:column; align-items:center; gap:2px; }
.obj-icon  { font-size:22px; }
.obj-count { font-size:17px; font-weight:700; color:#f0e6d2; }
.obj-name  { font-size:9px; color:#5b5a56; letter-spacing:1.5px; text-transform:uppercase; }
.obj-sep   { width:1px; background:rgba(200,170,110,0.15); margin:4px 0; }

/* prediction card */
.pred-card {
    background:linear-gradient(160deg,#0a1428,#091020);
    border:1px solid rgba(200,170,110,0.35); border-radius:4px;
    padding:28px; text-align:center;
    box-shadow:0 8px 40px rgba(0,0,0,0.7); position:relative;
}
.pred-card::before { content:''; position:absolute; top:0;left:0;right:0; height:2px;
    background:linear-gradient(90deg,transparent,#c8aa6e,transparent); }
.pred-prob   { font-size:58px; font-weight:700; font-family:'Cinzel',serif; margin:4px 0; }
.pred-vrd    { font-size:14px; font-weight:700; letter-spacing:3px; text-transform:uppercase; font-family:'Cinzel',serif; }
.pred-actual { font-size:11px; color:#5b5a56; margin-top:12px; letter-spacing:1px; }
.glow-win    { color:#0ac8b9; text-shadow:0 0 40px rgba(10,200,185,0.5); }
.glow-loss   { color:#e84057; text-shadow:0 0 40px rgba(232,64,87,0.5); }
.badge-ok  { display:inline-block; background:rgba(10,200,185,0.1); border:1px solid #0ac8b9; color:#0ac8b9; padding:5px 18px; border-radius:2px; font-size:10px; font-weight:700; letter-spacing:2px; margin-top:12px; text-transform:uppercase; }
.badge-ko  { display:inline-block; background:rgba(232,64,87,0.1);  border:1px solid #e84057; color:#e84057; padding:5px 18px; border-radius:2px; font-size:10px; font-weight:700; letter-spacing:2px; margin-top:12px; text-transform:uppercase; }

/* buttons */
.stButton>button {
    background:linear-gradient(180deg,#1e2328,#010a13);
    border:1px solid #c8aa6e !important; color:#c8aa6e !important;
    font-weight:700; text-transform:uppercase; letter-spacing:2px; font-size:13px;
    transition:all 0.25s; border-radius:2px; width:100%;
    box-shadow:0 0 20px rgba(200,170,110,0.1); font-family:'Cinzel',serif;
}
.stButton>button:hover {
    background:linear-gradient(180deg,#c8aa6e,#785a28) !important;
    color:#010a13 !important; box-shadow:0 0 40px rgba(200,170,110,0.5);
}
[data-testid="stDataFrame"] { border:1px solid rgba(200,170,110,0.2) !important; border-radius:4px !important; }
.stSlider>div>div>div { background:rgba(200,170,110,0.15) !important; }
.stSlider>div>div>div>div { background:#c8aa6e !important; }
/* expander */
details { border:1px solid rgba(200,170,110,0.2) !important; border-radius:4px !important; background:#0a1428 !important; }
summary { color:#c8aa6e !important; font-family:'Cinzel',serif; font-size:12px; letter-spacing:2px; }
</style>
"""

# ── Helpers ──────────────────────────────────────────────────────────────────────

def build_scoreboard(row: pd.Series) -> str:
    bk = int(row.get("blueKills", 0));   bd = int(row.get("blueDeaths", 0))
    ba = int(row.get("blueAssists", 0)); bg = int(row.get("blueTotalGold", 0))
    bc = int(row.get("blueTotalMinionsKilled", 0))
    gd = int(row.get("blueGoldDiff", 0)); xd = int(row.get("blueExperienceDiff", 0))
    bdr = int(row.get("blueDragons", 0)); bh = int(row.get("blueHeralds", 0)); bt = int(row.get("blueTowersDestroyed", 0))
    bkda = round((bk + ba) / max(1, bd), 2)
    bcsm = round(bc / 10, 1)

    # Red derived
    rk = bd; rd = bk; ra = int(row.get("redAssists", bk))
    rg = bg - gd; rc = int(row.get("redTotalMinionsKilled", 0)) or max(0, bc - int(gd / 15))
    rdr = int(row.get("redDragons", 0)); rh = int(row.get("redHeralds", 0)); rt = int(row.get("redTowersDestroyed", 0))
    rkda = round((rk + ra) / max(1, rd), 2)
    rcsm = round(rc / 10, 1)

    gd_s = f"+{gd:,}" if gd >= 0 else f"{gd:,}"; gd_c = "c-grn" if gd >= 0 else "c-red"
    xd_s = f"+{xd:,}" if xd >= 0 else f"{xd:,}"; xd_c = "c-grn" if xd >= 0 else "c-red"

    return f"""
<div class="sb-wrap">
  <div class="sb-totals">
    <div class="sb-totals-item"><div class="sb-tot-label">Blue Kills</div><div class="sb-tot-val sb-tot-blue">{bk}</div></div>
    <div class="sb-tot-mid">Total Kill</div>
    <div class="sb-totals-item"><div class="sb-tot-label">Red Kills</div><div class="sb-tot-val sb-tot-red">{rk}</div></div>
    <div style="width:1px;background:rgba(200,170,110,0.15);margin:0 16px;"></div>
    <div class="sb-totals-item"><div class="sb-tot-label">Blue Gold</div><div class="sb-tot-val sb-tot-blue">{bg:,}</div></div>
    <div class="sb-tot-mid">Total Gold</div>
    <div class="sb-totals-item"><div class="sb-tot-label">Red Gold</div><div class="sb-tot-val sb-tot-red">{rg:,}</div></div>
  </div>

  <div class="sb-team-hdr blue">🔵 &nbsp; Blue Team</div>
  <div class="sb-grid">
    <div class="sb-cell"><div class="sb-cell-label">Kills</div><div class="sb-cell-val c-blue">{bk}</div></div>
    <div class="sb-cell"><div class="sb-cell-label">Deaths</div><div class="sb-cell-val c-red">{bd}</div></div>
    <div class="sb-cell"><div class="sb-cell-label">Assists</div><div class="sb-cell-val c-blue">{ba}</div></div>
    <div class="sb-cell"><div class="sb-cell-label">KDA</div><div class="sb-cell-val c-teal">{bkda}</div></div>
    <div class="sb-cell"><div class="sb-cell-label">Gold</div><div class="sb-cell-val c-gold">{bg:,}</div></div>
    <div class="sb-cell"><div class="sb-cell-label">CS/min</div><div class="sb-cell-val c-blue">{bcsm}</div></div>
  </div>

  <div class="sb-team-hdr red">🔴 &nbsp; Red Team</div>
  <div class="sb-grid">
    <div class="sb-cell"><div class="sb-cell-label">Kills</div><div class="sb-cell-val c-red">{rk}</div></div>
    <div class="sb-cell"><div class="sb-cell-label">Deaths</div><div class="sb-cell-val c-blue">{rd}</div></div>
    <div class="sb-cell"><div class="sb-cell-label">Assists</div><div class="sb-cell-val c-red">{ra}</div></div>
    <div class="sb-cell"><div class="sb-cell-label">KDA</div><div class="sb-cell-val c-teal">{rkda}</div></div>
    <div class="sb-cell"><div class="sb-cell-label">Gold</div><div class="sb-cell-val c-gold">{rg:,}</div></div>
    <div class="sb-cell"><div class="sb-cell-label">CS/min</div><div class="sb-cell-val c-red">{rcsm}</div></div>
  </div>

  <div class="sb-diffs">
    <div class="diff-chip"><div class="diff-lbl">Gold Diff</div><div class="diff-val {gd_c}">{gd_s}</div></div>
    <div style="width:1px;background:rgba(200,170,110,0.1);"></div>
    <div class="diff-chip"><div class="diff-lbl">XP Diff</div><div class="diff-val {xd_c}">{xd_s}</div></div>
  </div>

  <div class="sb-objs">
    <div class="obj-b"><div class="obj-icon">🔵</div><div class="obj-count">{bdr}</div><div class="obj-name">B.Dragon</div></div>
    <div class="obj-b"><div class="obj-icon">🔴</div><div class="obj-count">{rdr}</div><div class="obj-name">R.Dragon</div></div>
    <div class="obj-sep"></div>
    <div class="obj-b"><div class="obj-icon">🔵</div><div class="obj-count">{bh}</div><div class="obj-name">B.Herald</div></div>
    <div class="obj-b"><div class="obj-icon">🔴</div><div class="obj-count">{rh}</div><div class="obj-name">R.Herald</div></div>
    <div class="obj-sep"></div>
    <div class="obj-b"><div class="obj-icon">🗼</div><div class="obj-count">{bt}</div><div class="obj-name">B.Tower</div></div>
    <div class="obj-b"><div class="obj-icon">🗼</div><div class="obj-count">{rt}</div><div class="obj-name">R.Tower</div></div>
  </div>
</div>"""


def build_pred_card(prob: float, pred: int, actual: int) -> str:
    win = pred == 1; ok = pred == actual
    vc = "glow-win" if win else "glow-loss"
    vt = "VICTORY PREDICTED" if win else "DEFEAT PREDICTED"
    badge = f'<span class="badge-ok">✦ CORRECT</span>' if ok else f'<span class="badge-ko">✦ INCORRECT</span>'
    at = "Victory" if actual == 1 else "Defeat"
    return f"""<div class="pred-card">
  <div style="font-size:10px;letter-spacing:3px;color:#5b5a56;text-transform:uppercase;font-family:'Cinzel',serif;">Blue Team Win Probability</div>
  <div class="pred-prob {vc}">{prob:.1%}</div>
  <div class="pred-vrd {vc}">{vt}</div>
  <div class="pred-actual">Actual outcome: <strong style="color:#f0e6d2">{at}</strong></div>
  {badge}
</div>"""


@st.cache_resource
def get_explainer(model_path: str):
    mdl = joblib.load(model_path)
    return shap.TreeExplainer(mdl), mdl


def shap_fig(shap_val, title: str):
    plt.style.use('dark_background')
    fig, _ = plt.subplots(figsize=(7, 3.8))
    fig.patch.set_facecolor('#0a1428')
    shap.plots.waterfall(shap_val, show=False)
    ax = plt.gca()
    ax.set_facecolor('#0a1428')
    ax.tick_params(colors='#a8b2bf')
    ax.xaxis.label.set_color('#a8b2bf')
    ax.yaxis.label.set_color('#a8b2bf')
    for text in ax.texts:
        text.set_color('#a8b2bf')
    for sp in ax.spines.values(): sp.set_edgecolor('none')
    ax.set_title(title, color='#c8aa6e', fontsize=10, pad=8)
    plt.tight_layout()
    return fig


# ── App ──────────────────────────────────────────────────────────────────────────

def build_app():
    st.set_page_config(page_title="LoL Victory Predictor", page_icon="⚔", layout="wide", initial_sidebar_state="collapsed")
    st.markdown(CSS, unsafe_allow_html=True)

    # Header
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if LOGO_PATH.exists():
            st.image(Image.open(LOGO_PATH), use_container_width=True)
        st.markdown("<div style='text-align:center;font-size:11px;letter-spacing:3px;color:#5b5a56;text-transform:uppercase;margin-top:-6px;'>10-Minute Victory Predictor · Hextech AI · Diamond Ranked</div>", unsafe_allow_html=True)
    st.markdown("<hr class='gold-line'>", unsafe_allow_html=True)

    # Model metrics (compact)
    mp = RESULTS_DIR / "model_metrics.csv"
    if mp.exists():
        with st.expander("⚡  MODEL PERFORMANCE METRICS", expanded=False):
            st.dataframe(pd.read_csv(mp), use_container_width=True)

    # SHAP beeswarm (compact expander)
    bs = PLOTS_DIR / "shap_beeswarm.png"
    if bs.exists():
        with st.expander("🧠  GLOBAL FEATURE IMPORTANCE (SHAP)", expanded=False):
            st.caption("How the model weighs each factor across all diamond-ranked games.")
            c1, c2, c3 = st.columns([1, 4, 1])
            with c2:
                st.image(Image.open(bs), use_container_width=True)

    st.markdown("<hr class='thin-line'>", unsafe_allow_html=True)

    # Live simulator
    st.markdown("<div class='sec-label'>⚔ Live Match Prediction Simulator</div>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#5b5a56;font-size:12px;letter-spacing:1px;margin-bottom:20px;'>Select a real high-diamond match · Inspect the 10-min snapshot · Execute the algorithm</p>", unsafe_allow_html=True)

    raw_feat, raw_tgt = load_raw_dataset()
    _, scaled_feat, _, _ = load_dataset_split()

    idx = st.slider("⚔ Match ID", 0, len(raw_feat) - 1, 0)
    row = raw_feat.iloc[idx]
    actual = int(raw_tgt.iloc[idx])

    st.markdown(build_scoreboard(row), unsafe_allow_html=True)

    xgb_path = MODELS_DIR / "xgboost.pkl"
    if st.button("⚡  EXECUTE HEXTECH PREDICTION ALGORITHM"):
        if xgb_path.exists():
            with st.spinner("Analysing match data..."):
                explainer, model = get_explainer(str(xgb_path))
                srow = scaled_feat.iloc[[idx]]
                pred = int(model.predict(srow)[0])
                prob = float(model.predict_proba(srow)[0][1])

            st.markdown(build_pred_card(prob, pred, actual), unsafe_allow_html=True)

            # Compact live SHAP waterfall inside expander
            st.markdown("<hr class='thin-line'>", unsafe_allow_html=True)
            with st.expander("🔍  DECISION SCAN — Why did the AI predict this? (SHAP)", expanded=True):
                st.caption("Step-by-step breakdown for this specific match. Blue bars push toward defeat, red bars toward victory.")
                with st.spinner("Computing explanation..."):
                    sv = explainer(srow)
                    fig = shap_fig(sv[0], f"Match #{idx} — Local SHAP Explanation")
                    c1, c2, c3 = st.columns([1, 5, 1])
                    with c2:
                        st.pyplot(fig)
                    plt.close(fig)
        else:
            st.error("SYSTEM OFFLINE — Run scripts/train.py first.")


if __name__ == "__main__":
    build_app()
