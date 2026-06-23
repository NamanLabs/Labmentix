"""
app.py — Shopper Spectrum | Dark/Light Mode + Always-visible sidebar info
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

# ── Session state ──────────────────────────────────────────────────────────
if "dark" not in st.session_state:
    st.session_state.dark = True

dark = st.session_state.dark

# ── Theme tokens ───────────────────────────────────────────────────────────
if dark:
    BG        = "#0f172a"
    SURFACE   = "#1e293b"
    SURFACE2  = "#0f172a"
    BORDER    = "#334155"
    TEXT      = "#f1f5f9"
    SUBTEXT   = "#94a3b8"
    MUTED     = "#475569"
    ACCENT    = "#6366f1"
    ACCENT2   = "#818cf8"
    SB_BG     = "#020617"
    SB_BORDER = "#1e293b"
    EMPTY_BG  = "#1e293b"
    INPUT_BG  = "#1e293b"
    TAG_BG    = "rgba(99,102,241,0.15)"
    TAG_BORDER= "rgba(99,102,241,0.4)"
    TAG_COLOR = "#818cf8"
else:
    BG        = "#f8fafc"
    SURFACE   = "#ffffff"
    SURFACE2  = "#f1f5f9"
    BORDER    = "#e2e8f0"
    TEXT      = "#0f172a"
    SUBTEXT   = "#475569"
    MUTED     = "#94a3b8"
    ACCENT    = "#6366f1"
    ACCENT2   = "#4f46e5"
    SB_BG     = "#0f172a"
    SB_BORDER = "#1e293b"
    EMPTY_BG  = "#f8fafc"
    INPUT_BG  = "#ffffff"
    TAG_BG    = "#eff6ff"
    TAG_BORDER= "#bfdbfe"
    TAG_COLOR = "#3b82f6"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* {{ font-family: 'Inter', sans-serif; box-sizing: border-box; }}
.stApp {{ background: {BG}; }}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding: 2rem 2.5rem; max-width: 1200px; }}

/* Sidebar */
[data-testid="stSidebar"] {{ background: {SB_BG}; border-right: 1px solid {SB_BORDER}; min-width: 260px; }}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown div {{ color: #94a3b8; }}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    background: transparent; border-bottom: 2px solid {BORDER}; gap: 0; padding: 0;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent !important; color: {MUTED} !important;
    font-size: 0.875rem !important; font-weight: 500 !important;
    padding: 0.65rem 1.3rem !important; border-bottom: 2px solid transparent !important;
    margin-bottom: -2px !important; border-radius: 0 !important;
}}
.stTabs [aria-selected="true"] {{
    color: {ACCENT} !important; border-bottom: 2px solid {ACCENT} !important;
    background: transparent !important;
}}

/* Inputs */
.stTextInput input, .stNumberInput input {{
    background: {INPUT_BG} !important; border: 1.5px solid {BORDER} !important;
    border-radius: 8px !important; color: {TEXT} !important;
    font-size: 0.875rem !important; padding: 0.6rem 1rem !important;
}}
.stTextInput input:focus, .stNumberInput input:focus {{
    border-color: {ACCENT} !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
}}
.stTextInput label, .stNumberInput label {{
    color: {SUBTEXT} !important; font-size: 0.8rem !important; font-weight: 500 !important;
}}

/* Button */
.stButton button {{
    background: {ACCENT} !important; color: white !important;
    border: none !important; border-radius: 8px !important;
    padding: 0.6rem 1.5rem !important; font-weight: 600 !important;
    font-size: 0.875rem !important; width: 100% !important;
    transition: all 0.2s !important;
}}
.stButton button:hover {{ background: #4f46e5 !important; transform: translateY(-1px) !important; }}

/* Scrollbar */
::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: {BG}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 10px; }}
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────
def build_similarity_matrix():
    df = pd.read_csv(DATA_PATH)
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df = df.drop_duplicates().dropna(subset=["CustomerID"])
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df = df[~df["InvoiceNo"].str.startswith("C")]
    dp = df[~df["StockCode"].str.upper().isin(ADMIN_CODES)]
    basket = dp.pivot_table(index="CustomerID", columns="StockCode",
                            values="Quantity", aggfunc="sum", fill_value=0)
    bbin = (basket > 0).astype(int)
    sim = cosine_similarity(bbin.T)
    sim_df = pd.DataFrame(sim, index=bbin.columns, columns=bbin.columns)
    joblib.dump(sim_df, os.path.join(SAVE_DIR, "item_similarity_matrix.pkl"))
    return sim_df

@st.cache_resource
def load_artifacts():
    req = ["rfm_scaler.pkl","segment_classifier.pkl",
           "segment_label_encoder.pkl","product_description_lookup.pkl"]
    if any(not os.path.exists(os.path.join(SAVE_DIR, f)) for f in req):
        return None
    scaler = joblib.load(os.path.join(SAVE_DIR, "rfm_scaler.pkl"))
    model  = joblib.load(os.path.join(SAVE_DIR, "segment_classifier.pkl"))
    le     = joblib.load(os.path.join(SAVE_DIR, "segment_label_encoder.pkl"))
    lookup = joblib.load(os.path.join(SAVE_DIR, "product_description_lookup.pkl"))
    sp = os.path.join(SAVE_DIR, "item_similarity_matrix.pkl")
    sim = joblib.load(sp) if os.path.exists(sp) else None
    if sim is None:
        with st.spinner("Building recommendation engine (~30s, first launch only)..."):
            sim = build_similarity_matrix()
    return {"scaler":scaler,"model":model,"le":le,"sim":sim,"lookup":lookup}

def do_recommend(q, arts):
    m = arts["lookup"][arts["lookup"].str.contains(q, case=False, na=False, regex=False)]
    if len(m) == 0: return None, None
    code = m.index[0]
    sims = arts["sim"][code].drop(code).sort_values(ascending=False).head(5)
    df = pd.DataFrame({"StockCode":sims.index,"Product":arts["lookup"].loc[sims.index].values,"Score":sims.values})
    return arts["lookup"].loc[code], df

def do_predict(r, f, m, arts):
    feat = pd.DataFrame({"Recency_log":[np.log1p(r)],"Frequency_log":[np.log1p(f)],"Monetary_log":[np.log1p(m)]})
    sc   = arts["scaler"].transform(feat)
    enc  = arts["model"].predict(sc)[0]
    prob = arts["model"].predict_proba(sc)[0]
    seg  = arts["le"].inverse_transform([enc])[0]
    pdf  = pd.DataFrame({"Segment":arts["le"].classes_,"Prob":prob}).sort_values("Prob",ascending=False).reset_index(drop=True)
    return seg, pdf

SEGS = {
    "High-Value": {"icon":"👑","color":"#d97706","bg":"#fef3c7","border":"#f59e0b",
                   "desc":"Your most valuable customers — recent, frequent and high-spending.",
                   "action":"🎯 Launch VIP loyalty program & early access rewards","bar":"#f59e0b"},
    "Regular":    {"icon":"⭐","color":"#1d4ed8","bg":"#dbeafe","border":"#3b82f6",
                   "desc":"Consistent, dependable buyers with solid purchase history.",
                   "action":"🎁 Introduce loyalty points and bundle deals","bar":"#3b82f6"},
    "Occasional": {"icon":"🔄","color":"#c2410c","bg":"#ffedd5","border":"#f97316",
                   "desc":"Infrequent buyers who respond well to targeted campaigns.",
                   "action":"📧 Send personalized re-engagement emails","bar":"#f97316"},
    "At-Risk":    {"icon":"⚠️","color":"#b91c1c","bg":"#fee2e2","border":"#ef4444",
                   "desc":"Customers who haven't purchased recently. Act fast.",
                   "action":"🚨 Send urgent win-back offer with heavy discount","bar":"#ef4444"},
}


# ── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo + mode toggle
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown(f"""
        <div style="padding:1.2rem 0 0.5rem 0;">
            <div style="font-size:1.15rem; font-weight:700; color:#f1f5f9;">🛒 Shopper Spectrum</div>
            <div style="font-size:0.72rem; color:#475569; margin-top:2px;">E-Commerce Intelligence</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("<div style='padding-top:1.2rem'>", unsafe_allow_html=True)
        if st.button("🌙" if not dark else "☀️", help="Toggle dark/light mode"):
            st.session_state.dark = not st.session_state.dark
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f"<hr style='border:none;border-top:1px solid #1e293b;margin:0.5rem 0 1rem 0'>", unsafe_allow_html=True)

    # Stats
    st.markdown("""
    <div style="font-size:0.65rem;color:#334155;font-weight:600;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:0.7rem;">
        Model Overview
    </div>""", unsafe_allow_html=True)

    stats = [("Customers Analyzed","4,338"),("Products Indexed","3,659"),
             ("Model Accuracy","93%"),("Algorithm","Random Forest")]
    for label, val in stats:
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;align-items:center;
             background:#1e293b;border-radius:8px;padding:0.5rem 0.75rem;margin-bottom:0.4rem;">
            <span style="color:#64748b;font-size:0.72rem;">{label}</span>
            <span style="color:#818cf8;font-size:0.8rem;font-weight:600;">{val}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"<hr style='border:none;border-top:1px solid #1e293b;margin:1rem 0'>", unsafe_allow_html=True)

    # Guide
    st.markdown("""
    <div style="font-size:0.65rem;color:#334155;font-weight:600;text-transform:uppercase;
         letter-spacing:1.5px;margin-bottom:0.7rem;">How to Use</div>""", unsafe_allow_html=True)

    steps = [
        ("Product Recommendations", "Type any product name and click Find Similar →"),
        ("Customer Segmentation", "Enter Recency, Frequency & Monetary values, then click Predict →"),
    ]
    for i, (title, desc) in enumerate(steps, 1):
        st.markdown(f"""
        <div style="display:flex;gap:0.7rem;margin-bottom:0.9rem;align-items:flex-start;">
            <div style="min-width:20px;height:20px;background:#6366f1;border-radius:50%;
                 display:flex;align-items:center;justify-content:center;
                 font-size:0.65rem;font-weight:700;color:white;margin-top:1px;">{i}</div>
            <div>
                <div style="color:#94a3b8;font-size:0.78rem;font-weight:600;">{title}</div>
                <div style="color:#475569;font-size:0.73rem;margin-top:2px;line-height:1.5;">{desc}</div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"<hr style='border:none;border-top:1px solid #1e293b;margin:0.5rem 0 1rem 0'>", unsafe_allow_html=True)

    # RFM Guide
    st.markdown("""
    <div style="font-size:0.65rem;color:#334155;font-weight:600;text-transform:uppercase;
         letter-spacing:1.5px;margin-bottom:0.7rem;">RFM Explained</div>""", unsafe_allow_html=True)

    rfm_items = [
        ("📅 Recency", "Days since last purchase", "Lower = more recent = better"),
        ("🔁 Frequency", "Number of orders placed", "Higher = more loyal = better"),
        ("💷 Monetary", "Total spend in £", "Higher = more valuable = better"),
    ]
    for icon_title, subtitle, tip in rfm_items:
        st.markdown(f"""
        <div style="margin-bottom:0.8rem;">
            <div style="color:#94a3b8;font-size:0.78rem;font-weight:600;">{icon_title}</div>
            <div style="color:#475569;font-size:0.72rem;">{subtitle}</div>
            <div style="color:#334155;font-size:0.7rem;font-style:italic;">{tip}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="position:absolute;bottom:1rem;left:1rem;right:1rem;text-align:center;
         color:#1e293b;font-size:0.7rem;">Built by Naman · Manipal University Jaipur</div>
    """, unsafe_allow_html=True)


# ── MAIN ──────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="margin-bottom:1.5rem;padding-bottom:1.2rem;border-bottom:1px solid {BORDER};">
    <h1 style="font-size:1.5rem;font-weight:700;color:{TEXT};margin:0;">Customer Intelligence Dashboard</h1>
    <p style="color:{SUBTEXT};font-size:0.875rem;margin:0.3rem 0 0 0;">
        Segment customers by RFM behavior &nbsp;·&nbsp; Discover product co-purchase patterns
    </p>
</div>""", unsafe_allow_html=True)

arts = load_artifacts()
if arts is None:
    st.error("Model files not found in `saved_models/`. Please run `python train_model.py` first.")
    st.stop()

tab1, tab2 = st.tabs(["🎯  Product Recommendations", "👥  Customer Segmentation"])

# ── TAB 1 ─────────────────────────────────────────────────────────────────
with tab1:
    col_q, col_btn = st.columns([5, 1])
    with col_q:
        query = st.text_input("Search product", placeholder="e.g.  WHITE HANGING HEART  ·  REGENCY CAKESTAND  ·  CERAMIC JAR", label_visibility="collapsed")
    with col_btn:
        st.markdown("<div style='margin-top:0.3rem'>", unsafe_allow_html=True)
        go = st.button("Find Similar →")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f"<div style='height:1rem'></div>", unsafe_allow_html=True)

    if go:
        if not query.strip():
            st.warning("Please enter a product name.")
        else:
            name, results = do_recommend(query, arts)
            if results is None:
                st.error(f"No product found matching **'{query}'**. Try a shorter keyword.")
            else:
                st.markdown(f"""
                <div style="display:inline-flex;align-items:center;gap:0.4rem;
                     background:{TAG_BG};border:1px solid {TAG_BORDER};
                     color:{TAG_COLOR};font-size:0.8rem;font-weight:500;
                     padding:0.35rem 0.8rem;border-radius:6px;margin-bottom:1rem;">
                    ✓ &nbsp; Showing results for: <b>&nbsp;{name}</b>
                </div>""", unsafe_allow_html=True)

                cols = st.columns(5)
                for i, (_, row) in enumerate(results.iterrows()):
                    pct = int(row["Score"] * 100)
                    with cols[i]:
                        st.markdown(f"""
                        <div style="background:{SURFACE};border:1px solid {BORDER};border-radius:12px;
                             padding:1rem;transition:all 0.2s;position:relative;height:180px;
                             display:flex;flex-direction:column;justify-content:space-between;">
                            <div>
                                <div style="font-size:0.65rem;color:{MUTED};font-weight:600;
                                     text-transform:uppercase;letter-spacing:0.5px;">#{i+1} match</div>
                                <div style="font-size:0.82rem;color:{TEXT};font-weight:500;
                                     line-height:1.4;margin:0.4rem 0;">{row['Product']}</div>
                            </div>
                            <div>
                                <div style="font-size:0.7rem;color:{MUTED};font-family:monospace;
                                     margin-bottom:0.4rem;">{row['StockCode']}</div>
                                <div style="height:3px;background:{BORDER};border-radius:10px;overflow:hidden;">
                                    <div style="height:100%;width:{pct}%;background:{ACCENT};border-radius:10px;"></div>
                                </div>
                                <div style="font-size:0.7rem;color:{ACCENT};font-weight:600;margin-top:0.3rem;">
                                    {row['Score']:.3f} similarity
                                </div>
                            </div>
                        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:{EMPTY_BG};border:1.5px dashed {BORDER};border-radius:12px;
             padding:3.5rem 2rem;text-align:center;">
            <div style="font-size:2rem;margin-bottom:0.5rem;">🔍</div>
            <div style="color:{SUBTEXT};font-size:0.875rem;">
                Type a product name above and click <b>Find Similar →</b>
            </div>
            <div style="color:{MUTED};font-size:0.78rem;margin-top:0.4rem;">
                Try: WHITE HANGING HEART &nbsp;·&nbsp; REGENCY CAKESTAND &nbsp;·&nbsp; CERAMIC TOP JAR
            </div>
        </div>""", unsafe_allow_html=True)


# ── TAB 2 ─────────────────────────────────────────────────────────────────
with tab2:
    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        st.markdown(f"""
        <div style="background:{SURFACE};border:1px solid {BORDER};border-radius:12px;
             padding:1.5rem;margin-bottom:0.75rem;">
            <div style="font-size:0.7rem;color:{MUTED};font-weight:600;text-transform:uppercase;
                 letter-spacing:1px;margin-bottom:1rem;">Enter Customer RFM Values</div>
        """, unsafe_allow_html=True)

        r = st.number_input("📅  Recency — days since last purchase", min_value=0, max_value=1000, value=30, step=1)
        f = st.number_input("🔁  Frequency — number of orders placed", min_value=1, max_value=500, value=5, step=1)
        m = st.number_input("💷  Monetary — total spend in £", min_value=0.0, max_value=500000.0, value=500.0, step=10.0)

        st.markdown("</div>", unsafe_allow_html=True)
        predict_btn = st.button("Predict Segment →")

    with col_out:
        if predict_btn:
            seg, pdf = do_predict(r, f, m, arts)
            cfg = SEGS[seg]

            st.markdown(f"""
            <div style="background:{cfg['bg']};border:1.5px solid {cfg['border']};
                 border-radius:12px;padding:1.8rem;margin-bottom:1rem;">
                <div style="font-size:2rem;">{cfg['icon']}</div>
                <div style="font-size:1.5rem;font-weight:700;color:{cfg['color']};margin:0.3rem 0;">{seg}</div>
                <div style="font-size:0.85rem;color:{cfg['color']}cc;line-height:1.6;">{cfg['desc']}</div>
                <div style="margin-top:1rem;background:rgba(0,0,0,0.06);border-radius:8px;
                     padding:0.6rem 0.9rem;font-size:0.78rem;font-weight:500;color:{cfg['color']};">
                    {cfg['action']}
                </div>
            </div>
            <div style="font-size:0.7rem;color:{MUTED};font-weight:600;text-transform:uppercase;
                 letter-spacing:1px;margin-bottom:0.7rem;">Prediction Confidence</div>
            """, unsafe_allow_html=True)

            for _, row in pdf.iterrows():
                pct = int(row["Prob"] * 100)
                bar_col = SEGS.get(row["Segment"], {}).get("bar", ACCENT)
                icon = SEGS.get(row["Segment"], {}).get("icon", "")
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:0.55rem;">
                    <div style="font-size:0.78rem;color:{SUBTEXT};width:105px;">{icon} {row['Segment']}</div>
                    <div style="flex:1;height:6px;background:{BORDER};border-radius:10px;overflow:hidden;">
                        <div style="height:100%;width:{pct}%;background:{bar_col};border-radius:10px;"></div>
                    </div>
                    <div style="font-size:0.75rem;color:{SUBTEXT};font-weight:600;width:32px;text-align:right;">{pct}%</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:{EMPTY_BG};border:1.5px dashed {BORDER};border-radius:12px;
                 padding:4rem 2rem;text-align:center;height:100%;">
                <div style="font-size:2rem;margin-bottom:0.5rem;">👥</div>
                <div style="color:{SUBTEXT};font-size:0.875rem;">
                    Enter RFM values and click <b>Predict Segment →</b>
                </div>
                <div style="color:{MUTED};font-size:0.78rem;margin-top:0.5rem;line-height:1.6;">
                    The model will predict whether the customer is<br>
                    High-Value · Regular · Occasional · At-Risk
                </div>
            </div>""", unsafe_allow_html=True)

