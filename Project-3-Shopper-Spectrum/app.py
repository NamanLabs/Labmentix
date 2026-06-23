"""
app.py — Shopper Spectrum | Clean Professional UI
"""

import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR  = os.path.join(BASE_DIR, "saved_models")
DATA_PATH = os.path.join(BASE_DIR, "online_retail.csv")
ADMIN_CODES = ["POST", "M", "C2", "DOT", "BANK CHARGES", "PADS", "CRUK"]

st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

.stApp { background: #f8fafc; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2.5rem 2.5rem 2rem 2.5rem; max-width: 1100px; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f172a;
    border-right: 1px solid #1e293b;
}
[data-testid="stSidebar"] * { color: #94a3b8 !important; }

.sidebar-logo {
    padding: 1.5rem 1.5rem 1rem 1.5rem;
    border-bottom: 1px solid #1e293b;
    margin-bottom: 1.5rem;
}
.sidebar-logo h2 {
    color: #f1f5f9 !important;
    font-size: 1.2rem !important;
    font-weight: 700 !important;
    margin: 0 !important;
}
.sidebar-logo p {
    color: #475569 !important;
    font-size: 0.75rem !important;
    margin: 0.2rem 0 0 0 !important;
}

.sidebar-section {
    padding: 0 1.5rem;
    margin-bottom: 1.5rem;
}
.sidebar-section-title {
    color: #334155 !important;
    font-size: 0.65rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
    margin-bottom: 0.8rem !important;
}
.guide-item {
    display: flex;
    gap: 0.8rem;
    margin-bottom: 1rem;
    align-items: flex-start;
}
.guide-num {
    background: #1e293b;
    color: #6366f1 !important;
    font-size: 0.7rem;
    font-weight: 700;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    margin-top: 1px;
}
.guide-text {
    color: #64748b !important;
    font-size: 0.78rem !important;
    line-height: 1.5 !important;
}

.stat-pill {
    background: #1e293b;
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
    margin-bottom: 0.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.stat-pill-label { color: #475569 !important; font-size: 0.72rem !important; }
.stat-pill-value { color: #6366f1 !important; font-size: 0.85rem !important; font-weight: 600 !important; }

/* Main area */
.page-header {
    margin-bottom: 2rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid #e2e8f0;
}
.page-header h1 {
    font-size: 1.6rem;
    font-weight: 700;
    color: #0f172a;
    margin: 0;
}
.page-header p {
    color: #64748b;
    font-size: 0.875rem;
    margin: 0.3rem 0 0 0;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 2px solid #e2e8f0;
    gap: 0;
    padding: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #94a3b8 !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    padding: 0.7rem 1.2rem !important;
    border-bottom: 2px solid transparent !important;
    margin-bottom: -2px !important;
    border-radius: 0 !important;
}
.stTabs [aria-selected="true"] {
    color: #6366f1 !important;
    border-bottom: 2px solid #6366f1 !important;
    background: transparent !important;
}

/* Input */
.stTextInput input, .stNumberInput input {
    background: white !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 8px !important;
    color: #0f172a !important;
    font-size: 0.875rem !important;
    padding: 0.6rem 1rem !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
}
.stTextInput label, .stNumberInput label {
    color: #374151 !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
}

/* Button */
.stButton button {
    background: #6366f1 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.5rem !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    width: 100% !important;
    transition: background 0.2s !important;
}
.stButton button:hover { background: #4f46e5 !important; }

/* Product cards */
.product-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 0.75rem;
    margin-top: 1.25rem;
}
.pcard {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem;
    transition: all 0.2s;
    position: relative;
}
.pcard:hover {
    border-color: #6366f1;
    box-shadow: 0 4px 20px rgba(99,102,241,0.12);
    transform: translateY(-2px);
}
.pcard-rank {
    font-size: 0.65rem;
    color: #94a3b8;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.pcard-name {
    font-size: 0.82rem;
    color: #1e293b;
    font-weight: 500;
    line-height: 1.4;
    margin: 0.4rem 0;
    min-height: 60px;
}
.pcard-code {
    font-size: 0.7rem;
    color: #94a3b8;
    font-family: monospace;
    margin-bottom: 0.5rem;
}
.pcard-score {
    font-size: 0.72rem;
    color: #6366f1;
    font-weight: 600;
}
.score-bar {
    height: 3px;
    background: #f1f5f9;
    border-radius: 10px;
    margin-top: 0.3rem;
    overflow: hidden;
}
.score-fill {
    height: 100%;
    background: #6366f1;
    border-radius: 10px;
}

/* Result box */
.match-tag {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    color: #3b82f6;
    font-size: 0.8rem;
    font-weight: 500;
    padding: 0.4rem 0.8rem;
    border-radius: 6px;
    margin-bottom: 1rem;
}

/* Segment result */
.seg-box {
    border-radius: 12px;
    padding: 1.8rem;
    margin-bottom: 1rem;
}
.seg-icon { font-size: 2.2rem; }
.seg-name { font-size: 1.5rem; font-weight: 700; margin: 0.3rem 0; }
.seg-desc { font-size: 0.85rem; line-height: 1.6; opacity: 0.9; margin-top: 0.3rem; }
.seg-action {
    margin-top: 1rem;
    padding: 0.6rem 0.9rem;
    background: rgba(255,255,255,0.15);
    border-radius: 8px;
    font-size: 0.78rem;
    font-weight: 500;
}

/* Confidence */
.conf-section { margin-top: 1rem; }
.conf-title {
    font-size: 0.75rem;
    color: #64748b;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.8rem;
}
.conf-item {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 0.6rem;
}
.conf-lbl { font-size: 0.8rem; color: #475569; width: 100px; }
.conf-track { flex:1; height:6px; background:#f1f5f9; border-radius:10px; overflow:hidden; }
.conf-fill { height:100%; border-radius:10px; }
.conf-pct { font-size:0.75rem; color:#64748b; font-weight:600; width:32px; text-align:right; }

/* RFM inputs container */
.rfm-wrap {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* Empty state */
.empty-state {
    background: #f8fafc;
    border: 1.5px dashed #e2e8f0;
    border-radius: 12px;
    padding: 3rem 2rem;
    text-align: center;
    color: #94a3b8;
    font-size: 0.875rem;
}
.empty-state .emoji { font-size: 2rem; margin-bottom: 0.5rem; }

/* Divider */
hr { border: none; border-top: 1px solid #e2e8f0; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
def build_similarity_matrix():
    df = pd.read_csv(DATA_PATH)
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df = df.drop_duplicates()
    df = df.dropna(subset=["CustomerID"])
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df = df[~df["InvoiceNo"].str.startswith("C")]
    df_p = df[~df["StockCode"].str.upper().isin(ADMIN_CODES)]
    basket = df_p.pivot_table(index="CustomerID", columns="StockCode",
                               values="Quantity", aggfunc="sum", fill_value=0)
    bbin = (basket > 0).astype(int)
    sim = cosine_similarity(bbin.T)
    item_sim_df = pd.DataFrame(sim, index=bbin.columns, columns=bbin.columns)
    joblib.dump(item_sim_df, os.path.join(SAVE_DIR, "item_similarity_matrix.pkl"))
    return item_sim_df


@st.cache_resource
def load_artifacts():
    req = ["rfm_scaler.pkl","segment_classifier.pkl",
           "segment_label_encoder.pkl","product_description_lookup.pkl"]
    if any(not os.path.exists(os.path.join(SAVE_DIR, f)) for f in req):
        return None
    scaler   = joblib.load(os.path.join(SAVE_DIR, "rfm_scaler.pkl"))
    model    = joblib.load(os.path.join(SAVE_DIR, "segment_classifier.pkl"))
    le       = joblib.load(os.path.join(SAVE_DIR, "segment_label_encoder.pkl"))
    lookup   = joblib.load(os.path.join(SAVE_DIR, "product_description_lookup.pkl"))
    sp = os.path.join(SAVE_DIR, "item_similarity_matrix.pkl")
    if os.path.exists(sp):
        sim = joblib.load(sp)
    else:
        with st.spinner("Building recommendation engine (first launch only, ~30s)..."):
            sim = build_similarity_matrix()
    return {"scaler":scaler,"model":model,"le":le,"sim":sim,"lookup":lookup}


SEGS = {
    "High-Value": {"icon":"👑","color":"#92400e","bg":"#fffbeb","border":"#f59e0b",
                   "desc":"Your most valuable customers — recent, frequent and high-spending.",
                   "action":"🎯 Launch VIP loyalty program & early access rewards"},
    "Regular":    {"icon":"⭐","color":"#1e40af","bg":"#eff6ff","border":"#3b82f6",
                   "desc":"Consistent, dependable buyers with solid purchase history.",
                   "action":"🎁 Introduce loyalty points and bundle deals"},
    "Occasional": {"icon":"🔄","color":"#9a3412","bg":"#fff7ed","border":"#f97316",
                   "desc":"Infrequent buyers who respond well to targeted campaigns.",
                   "action":"📧 Send personalized re-engagement emails"},
    "At-Risk":    {"icon":"⚠️","color":"#991b1b","bg":"#fef2f2","border":"#ef4444",
                   "desc":"Customers who haven't purchased in a long time. Act fast.",
                   "action":"🚨 Send urgent win-back offer with heavy discount"},
}
SEG_COLORS = {"High-Value":"#f59e0b","Regular":"#3b82f6","Occasional":"#f97316","At-Risk":"#ef4444"}


def do_recommend(q, arts):
    matches = arts["lookup"][arts["lookup"].str.contains(q, case=False, na=False, regex=False)]
    if len(matches) == 0:
        return None, None
    code = matches.index[0]
    sims = arts["sim"][code].drop(code).sort_values(ascending=False).head(5)
    df = pd.DataFrame({"StockCode":sims.index, "Product":arts["lookup"].loc[sims.index].values, "Score":sims.values})
    return arts["lookup"].loc[code], df


def do_predict(r, f, m, arts):
    feat = pd.DataFrame({"Recency_log":[np.log1p(r)],"Frequency_log":[np.log1p(f)],"Monetary_log":[np.log1p(m)]})
    scaled = arts["scaler"].transform(feat)
    enc = arts["model"].predict(scaled)[0]
    proba = arts["model"].predict_proba(scaled)[0]
    seg = arts["le"].inverse_transform([enc])[0]
    pdf = pd.DataFrame({"Segment":arts["le"].classes_,"Prob":proba}).sort_values("Prob",ascending=False).reset_index(drop=True)
    return seg, pdf


# --------------------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <h2>🛒 Shopper Spectrum</h2>
        <p>E-Commerce Intelligence Platform</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section">
        <div class="sidebar-section-title">How to use</div>
        <div class="guide-item">
            <div class="guide-num">1</div>
            <div class="guide-text">Go to <b style="color:#94a3b8">Product Recommendations</b> tab and type any product name</div>
        </div>
        <div class="guide-item">
            <div class="guide-num">2</div>
            <div class="guide-text">Click <b style="color:#94a3b8">Find Similar</b> to see 5 products customers bought together</div>
        </div>
        <div class="guide-item">
            <div class="guide-num">3</div>
            <div class="guide-text">Switch to <b style="color:#94a3b8">Customer Segmentation</b> tab</div>
        </div>
        <div class="guide-item">
            <div class="guide-num">4</div>
            <div class="guide-text">Enter Recency, Frequency and Monetary values, then click <b style="color:#94a3b8">Predict</b></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1e293b; margin:0 1.5rem 1.5rem 1.5rem;'>", unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section">
        <div class="sidebar-section-title">Model Info</div>
        <div class="stat-pill"><span class="stat-pill-label">Customers</span><span class="stat-pill-value">4,338</span></div>
        <div class="stat-pill"><span class="stat-pill-label">Products</span><span class="stat-pill-value">3,659</span></div>
        <div class="stat-pill"><span class="stat-pill-label">Accuracy</span><span class="stat-pill-value">93%</span></div>
        <div class="stat-pill"><span class="stat-pill-label">Algorithm</span><span class="stat-pill-value">Random Forest</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1e293b; margin:0 1.5rem 1.5rem 1.5rem;'>", unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section">
        <div class="sidebar-section-title">RFM Guide</div>
        <div class="guide-item">
            <div class="guide-text"><b style="color:#94a3b8">Recency</b> — Days since last purchase. Lower is better.</div>
        </div>
        <div class="guide-item">
            <div class="guide-text"><b style="color:#94a3b8">Frequency</b> — Total number of purchases. Higher is better.</div>
        </div>
        <div class="guide-item">
            <div class="guide-text"><b style="color:#94a3b8">Monetary</b> — Total spend in £. Higher is better.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
st.markdown("""
<div class="page-header">
    <h1>Customer Intelligence Dashboard</h1>
    <p>Segment customers by RFM behavior · Discover product co-purchase patterns</p>
</div>
""", unsafe_allow_html=True)

arts = load_artifacts()
if arts is None:
    st.error("Model files not found. Please run `python train_model.py` first.")
    st.stop()

tab1, tab2 = st.tabs(["🎯  Product Recommendations", "👥  Customer Segmentation"])

# ── TAB 1 ────────────────────────────────────────────────────────────────
with tab1:
    col_q, col_btn = st.columns([5, 1])
    with col_q:
        query = st.text_input("Product name", placeholder="e.g.  WHITE HANGING HEART   or   REGENCY CAKESTAND", label_visibility="collapsed")
    with col_btn:
        st.markdown("<div style='margin-top:1.9rem'>", unsafe_allow_html=True)
        go = st.button("Find Similar →")
        st.markdown("</div>", unsafe_allow_html=True)

    if go:
        if not query.strip():
            st.warning("Enter a product name to search.")
        else:
            name, results = do_recommend(query, arts)
            if results is None:
                st.error(f"No product found matching **'{query}'**. Try a shorter keyword.")
            else:
                st.markdown(f'<div class="match-tag">✓ &nbsp; Results for: <b>&nbsp;{name}</b></div>', unsafe_allow_html=True)
                cols = st.columns(5)
                for i, (_, row) in enumerate(results.iterrows()):
                    pct = int(row["Score"] * 100)
                    with cols[i]:
                        st.markdown(f"""
                        <div class="pcard">
                            <div class="pcard-rank">#{i+1}</div>
                            <div class="pcard-name">{row['Product']}</div>
                            <div class="pcard-code">{row['StockCode']}</div>
                            <div class="pcard-score">{row['Score']:.3f} match</div>
                            <div class="score-bar"><div class="score-fill" style="width:{pct}%"></div></div>
                        </div>
                        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="emoji">🔍</div>
            Type a product name above and click <b>Find Similar</b>
        </div>
        """, unsafe_allow_html=True)


# ── TAB 2 ────────────────────────────────────────────────────────────────
with tab2:
    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        st.markdown('<div class="rfm-wrap">', unsafe_allow_html=True)
        r = st.number_input("📅  Recency (days since last purchase)", min_value=0, max_value=1000, value=30, step=1)
        f = st.number_input("🔁  Frequency (number of orders)", min_value=1, max_value=500, value=5, step=1)
        m = st.number_input("💷  Monetary (total spend in £)", min_value=0.0, max_value=500000.0, value=500.0, step=10.0)
        st.markdown('</div>', unsafe_allow_html=True)
        predict_btn = st.button("Predict Segment →")

    with col_out:
        if predict_btn:
            seg, pdf = do_predict(r, f, m, arts)
            cfg = SEGS[seg]
            st.markdown(f"""
            <div class="seg-box" style="background:{cfg['bg']}; border:1.5px solid {cfg['border']};">
                <div class="seg-icon">{cfg['icon']}</div>
                <div class="seg-name" style="color:{cfg['color']};">{seg}</div>
                <div class="seg-desc" style="color:{cfg['color']}99;">{cfg['desc']}</div>
                <div class="seg-action" style="color:{cfg['color']};">{cfg['action']}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="conf-section"><div class="conf-title">Prediction Confidence</div>', unsafe_allow_html=True)
            for _, row in pdf.iterrows():
                pct = int(row["Prob"] * 100)
                col = SEG_COLORS.get(row["Segment"], "#6366f1")
                icon = SEGS.get(row["Segment"], {}).get("icon", "")
                st.markdown(f"""
                <div class="conf-item">
                    <div class="conf-lbl">{icon} {row['Segment']}</div>
                    <div class="conf-track"><div class="conf-fill" style="width:{pct}%;background:{col};"></div></div>
                    <div class="conf-pct">{pct}%</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="empty-state" style="margin-top:0.5rem;">
                <div class="emoji">👥</div>
                Enter RFM values and click <b>Predict Segment</b>
            </div>
            """, unsafe_allow_html=True)

