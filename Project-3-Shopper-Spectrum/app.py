"""
app.py — Shopper Spectrum | Always-visible theme toggle + guide
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

st.set_page_config(page_title="Shopper Spectrum", page_icon="🛒",
                   layout="wide", initial_sidebar_state="collapsed")

if "dark" not in st.session_state:
    st.session_state.dark = True

dark = st.session_state.dark

# ── Theme tokens ──────────────────────────────────────────────────────────
if dark:
    BG, SURFACE, SURFACE2 = "#0f172a", "#1e293b", "#0f172a"
    BORDER, TEXT, SUBTEXT  = "#334155", "#f1f5f9", "#94a3b8"
    MUTED, ACCENT          = "#475569", "#6366f1"
    INPUT_BG, EMPTY_BG     = "#1e293b", "#1e293b"
    TAG_BG, TAG_COLOR      = "rgba(99,102,241,0.15)", "#818cf8"
    TAG_BORDER             = "rgba(99,102,241,0.4)"
    CARD_HOVER             = "#263348"
    TOGGLE_BG              = "#1e293b"
    TOGGLE_COLOR           = "#f1f5f9"
    TOGGLE_ICON            = "☀️"
    TOGGLE_LABEL           = "Light Mode"
else:
    BG, SURFACE, SURFACE2  = "#f8fafc", "#ffffff", "#f1f5f9"
    BORDER, TEXT, SUBTEXT   = "#e2e8f0", "#0f172a", "#475569"
    MUTED, ACCENT           = "#94a3b8", "#6366f1"
    INPUT_BG, EMPTY_BG      = "#ffffff", "#f8fafc"
    TAG_BG, TAG_COLOR       = "#eff6ff", "#3b82f6"
    TAG_BORDER              = "#bfdbfe"
    CARD_HOVER              = "#f8fafc"
    TOGGLE_BG               = "#0f172a"
    TOGGLE_COLOR            = "#ffffff"
    TOGGLE_ICON             = "🌙"
    TOGGLE_LABEL            = "Dark Mode"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* {{ font-family:'Inter',sans-serif; box-sizing:border-box; }}
.stApp {{ background:{BG}; }}
#MainMenu, footer, header {{ visibility:hidden; }}
.block-container {{ padding:1.5rem 2.5rem 2rem 2.5rem; max-width:1200px; }}

/* Hide sidebar completely */
[data-testid="stSidebar"] {{ display:none; }}
[data-testid="collapsedControl"] {{ display:none; }}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    background:transparent; border-bottom:2px solid {BORDER}; gap:0; padding:0;
}}
.stTabs [data-baseweb="tab"] {{
    background:transparent !important; color:{MUTED} !important;
    font-size:0.875rem !important; font-weight:500 !important;
    padding:0.65rem 1.3rem !important; border-bottom:2px solid transparent !important;
    margin-bottom:-2px !important; border-radius:0 !important;
}}
.stTabs [aria-selected="true"] {{
    color:{ACCENT} !important; border-bottom:2px solid {ACCENT} !important;
    background:transparent !important;
}}

/* Inputs */
.stTextInput input, .stNumberInput input {{
    background:{INPUT_BG} !important; border:1.5px solid {BORDER} !important;
    border-radius:8px !important; color:{TEXT} !important;
    font-size:0.875rem !important; padding:0.6rem 1rem !important;
}}
.stTextInput input:focus, .stNumberInput input:focus {{
    border-color:{ACCENT} !important;
    box-shadow:0 0 0 3px rgba(99,102,241,0.12) !important;
}}
.stTextInput label, .stNumberInput label {{
    color:{SUBTEXT} !important; font-size:0.8rem !important; font-weight:500 !important;
}}

/* Buttons */
.stButton > button {{
    border:none !important; border-radius:8px !important;
    font-weight:600 !important; font-size:0.875rem !important;
    transition:all 0.2s !important;
}}

/* Scrollbar */
::-webkit-scrollbar {{ width:5px; }}
::-webkit-scrollbar-track {{ background:{BG}; }}
::-webkit-scrollbar-thumb {{ background:{BORDER}; border-radius:10px; }}
</style>
""", unsafe_allow_html=True)


# ── Artifacts ─────────────────────────────────────────────────────────────
def build_similarity_matrix():
    df = pd.read_csv(DATA_PATH)
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df = df.drop_duplicates().dropna(subset=["CustomerID"])
    df = df[(df["Quantity"]>0)&(df["UnitPrice"]>0)]
    df = df[~df["InvoiceNo"].str.startswith("C")]
    dp = df[~df["StockCode"].str.upper().isin(ADMIN_CODES)]
    b  = dp.pivot_table(index="CustomerID",columns="StockCode",values="Quantity",aggfunc="sum",fill_value=0)
    bb = (b>0).astype(int)
    s  = cosine_similarity(bb.T)
    sd = pd.DataFrame(s,index=bb.columns,columns=bb.columns)
    joblib.dump(sd, os.path.join(SAVE_DIR,"item_similarity_matrix.pkl"))
    return sd

@st.cache_resource
def load_artifacts():
    req = ["rfm_scaler.pkl","segment_classifier.pkl","segment_label_encoder.pkl","product_description_lookup.pkl"]
    if any(not os.path.exists(os.path.join(SAVE_DIR,f)) for f in req): return None
    scaler = joblib.load(os.path.join(SAVE_DIR,"rfm_scaler.pkl"))
    model  = joblib.load(os.path.join(SAVE_DIR,"segment_classifier.pkl"))
    le     = joblib.load(os.path.join(SAVE_DIR,"segment_label_encoder.pkl"))
    lk     = joblib.load(os.path.join(SAVE_DIR,"product_description_lookup.pkl"))
    sp     = os.path.join(SAVE_DIR,"item_similarity_matrix.pkl")
    sim    = joblib.load(sp) if os.path.exists(sp) else None
    if sim is None:
        with st.spinner("Building recommendation engine (first launch, ~30s)..."): sim=build_similarity_matrix()
    return {"scaler":scaler,"model":model,"le":le,"sim":sim,"lookup":lk}

def do_recommend(q,arts):
    m = arts["lookup"][arts["lookup"].str.contains(q,case=False,na=False,regex=False)]
    if not len(m): return None,None
    c = m.index[0]
    s = arts["sim"][c].drop(c).sort_values(ascending=False).head(5)
    return arts["lookup"].loc[c], pd.DataFrame({"StockCode":s.index,"Product":arts["lookup"].loc[s.index].values,"Score":s.values})

def do_predict(r,f,m,arts):
    feat = pd.DataFrame({"Recency_log":[np.log1p(r)],"Frequency_log":[np.log1p(f)],"Monetary_log":[np.log1p(m)]})
    sc   = arts["scaler"].transform(feat)
    enc  = arts["model"].predict(sc)[0]
    prob = arts["model"].predict_proba(sc)[0]
    seg  = arts["le"].inverse_transform([enc])[0]
    pdf  = pd.DataFrame({"Segment":arts["le"].classes_,"Prob":prob}).sort_values("Prob",ascending=False).reset_index(drop=True)
    return seg, pdf

SEGS = {
    "High-Value":{"icon":"👑","color":"#d97706","bg":"#fef3c7","border":"#f59e0b","bar":"#f59e0b",
                  "desc":"Your most valuable customers — recent, frequent and high-spending.",
                  "action":"🎯 Launch VIP loyalty program & early access rewards"},
    "Regular":   {"icon":"⭐","color":"#1d4ed8","bg":"#dbeafe","border":"#3b82f6","bar":"#3b82f6",
                  "desc":"Consistent, dependable buyers with solid purchase history.",
                  "action":"🎁 Introduce loyalty points and bundle deals"},
    "Occasional":{"icon":"🔄","color":"#c2410c","bg":"#ffedd5","border":"#f97316","bar":"#f97316",
                  "desc":"Infrequent buyers who respond well to targeted campaigns.",
                  "action":"📧 Send personalized re-engagement emails"},
    "At-Risk":   {"icon":"⚠️","color":"#b91c1c","bg":"#fee2e2","border":"#ef4444","bar":"#ef4444",
                  "desc":"Customers who haven't purchased recently. Act fast.",
                  "action":"🚨 Send urgent win-back offer with heavy discount"},
}


# ══════════════════════════════════════════════════════════════════════════
# TOP NAVBAR — always visible, theme toggle here
# ══════════════════════════════════════════════════════════════════════════
nav_left, nav_right = st.columns([5, 1])

with nav_left:
    st.markdown(f"""
    <div style="padding:0.5rem 0 1rem 0;">
        <div style="font-size:1.4rem;font-weight:700;color:{TEXT};">🛒 Shopper Spectrum</div>
        <div style="font-size:0.8rem;color:{MUTED};margin-top:1px;">
            Customer Segmentation &nbsp;·&nbsp; Product Recommendations &nbsp;·&nbsp; E-Commerce Intelligence
        </div>
    </div>""", unsafe_allow_html=True)

with nav_right:
    st.markdown("<div style='padding-top:0.6rem'>", unsafe_allow_html=True)
    if st.button(f"{TOGGLE_ICON} {TOGGLE_LABEL}", help="Switch theme"):
        st.session_state.dark = not dark
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(f"<hr style='border:none;border-top:1px solid {BORDER};margin:0 0 1.5rem 0'>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# STATS ROW — always visible below navbar
# ══════════════════════════════════════════════════════════════════════════
s1,s2,s3,s4 = st.columns(4)
stats = [
    (s1,"4,338","Customers Analyzed","👥"),
    (s2,"93%","Model Accuracy","🎯"),
    (s3,"3,659","Products Indexed","📦"),
    (s4,"Random Forest","Algorithm","🤖"),
]
for col, val, label, icon in stats:
    with col:
        st.markdown(f"""
        <div style="background:{SURFACE};border:1px solid {BORDER};border-radius:10px;
             padding:0.9rem 1.1rem;margin-bottom:1.5rem;">
            <div style="font-size:0.7rem;color:{MUTED};font-weight:600;text-transform:uppercase;
                 letter-spacing:0.5px;">{icon} &nbsp;{label}</div>
            <div style="font-size:1.2rem;font-weight:700;color:{TEXT};margin-top:0.3rem;">{val}</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════
arts = load_artifacts()
if arts is None:
    st.error("Model files not found. Please run `python train_model.py` first.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["🎯  Product Recommendations", "👥  Customer Segmentation", "📖  How to Use"])

# ── TAB 1: Recommendations ────────────────────────────────────────────────
with tab1:
    st.markdown(f"""
    <div style="margin-bottom:1rem;">
        <div style="font-size:0.95rem;font-weight:600;color:{TEXT};">Find Similar Products</div>
        <div style="font-size:0.8rem;color:{MUTED};margin-top:2px;">
            Based on real purchase co-occurrence across 4,338 customers
        </div>
    </div>""", unsafe_allow_html=True)

    cq, cb = st.columns([5,1])
    with cq:
        query = st.text_input("", placeholder="🔍  Try: WHITE HANGING HEART  ·  REGENCY CAKESTAND  ·  CERAMIC JAR", label_visibility="collapsed")
    with cb:
        st.markdown("<div style='margin-top:0.2rem'>",unsafe_allow_html=True)
        go = st.button("Search →", use_container_width=True)
        st.markdown("</div>",unsafe_allow_html=True)

    if go:
        if not query.strip():
            st.warning("Enter a product name.")
        else:
            name, results = do_recommend(query, arts)
            if results is None:
                st.error(f"No product found for **'{query}'**. Try a shorter keyword.")
            else:
                st.markdown(f"""
                <div style="display:inline-flex;align-items:center;gap:0.4rem;background:{TAG_BG};
                     border:1px solid {TAG_BORDER};color:{TAG_COLOR};font-size:0.8rem;font-weight:500;
                     padding:0.35rem 0.8rem;border-radius:6px;margin:0.8rem 0 1rem 0;">
                    ✓ Results for: <b>&nbsp;{name}</b>
                </div>""", unsafe_allow_html=True)

                cols = st.columns(5)
                for i,(_, row) in enumerate(results.iterrows()):
                    pct = int(row["Score"]*100)
                    with cols[i]:
                        st.markdown(f"""
                        <div style="background:{SURFACE};border:1px solid {BORDER};border-radius:12px;
                             padding:1rem;height:185px;display:flex;flex-direction:column;
                             justify-content:space-between;">
                            <div>
                                <div style="font-size:0.65rem;color:{MUTED};font-weight:600;
                                     text-transform:uppercase;letter-spacing:0.5px;">#{i+1}</div>
                                <div style="font-size:0.82rem;color:{TEXT};font-weight:500;
                                     line-height:1.4;margin:0.35rem 0;">{row['Product']}</div>
                            </div>
                            <div>
                                <div style="font-size:0.68rem;color:{MUTED};font-family:monospace;
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
             padding:3rem 2rem;text-align:center;margin-top:0.5rem;">
            <div style="font-size:2rem;margin-bottom:0.5rem;">🔍</div>
            <div style="color:{SUBTEXT};font-size:0.9rem;font-weight:500;">Search for a product above</div>
            <div style="color:{MUTED};font-size:0.78rem;margin-top:0.4rem;">
                e.g. &nbsp; WHITE HANGING HEART &nbsp;·&nbsp; REGENCY CAKESTAND &nbsp;·&nbsp; CERAMIC TOP JAR
            </div>
        </div>""", unsafe_allow_html=True)


# ── TAB 2: Segmentation ───────────────────────────────────────────────────
with tab2:
    st.markdown(f"""
    <div style="margin-bottom:1rem;">
        <div style="font-size:0.95rem;font-weight:600;color:{TEXT};">Predict Customer Segment</div>
        <div style="font-size:0.8rem;color:{MUTED};margin-top:2px;">
            Enter RFM values to predict segment using Random Forest (93% accuracy)
        </div>
    </div>""", unsafe_allow_html=True)

    ci, co = st.columns([1,1], gap="large")

    with ci:
        st.markdown(f"""
        <div style="background:{SURFACE};border:1px solid {BORDER};border-radius:12px;padding:1.5rem 1.5rem 0.5rem 1.5rem;">
        """, unsafe_allow_html=True)
        r = st.number_input("📅  Recency (days since last purchase)", min_value=0, max_value=1000, value=30)
        f = st.number_input("🔁  Frequency (number of orders)", min_value=1, max_value=500, value=5)
        m = st.number_input("💷  Monetary (total spend in £)", min_value=0.0, max_value=500000.0, value=500.0, step=10.0)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div style='margin-top:0.75rem'>",unsafe_allow_html=True)
        pb = st.button("⚡  Predict Segment", use_container_width=True)
        st.markdown("</div>",unsafe_allow_html=True)

    with co:
        if pb:
            seg, pdf = do_predict(r,f,m,arts)
            cfg = SEGS[seg]
            st.markdown(f"""
            <div style="background:{cfg['bg']};border:1.5px solid {cfg['border']};
                 border-radius:12px;padding:1.5rem;margin-bottom:1rem;">
                <div style="font-size:1.8rem;">{cfg['icon']}</div>
                <div style="font-size:1.4rem;font-weight:700;color:{cfg['color']};margin:0.3rem 0;">{seg}</div>
                <div style="font-size:0.83rem;color:{cfg['color']}bb;line-height:1.6;">{cfg['desc']}</div>
                <div style="margin-top:0.9rem;background:rgba(0,0,0,0.07);border-radius:8px;
                     padding:0.55rem 0.85rem;font-size:0.78rem;font-weight:500;color:{cfg['color']};">
                    {cfg['action']}
                </div>
            </div>
            <div style="font-size:0.7rem;color:{MUTED};font-weight:600;text-transform:uppercase;
                 letter-spacing:1px;margin-bottom:0.6rem;">Prediction Confidence</div>
            """, unsafe_allow_html=True)
            for _, row in pdf.iterrows():
                pct = int(row["Prob"]*100)
                bc  = SEGS.get(row["Segment"],{}).get("bar",ACCENT)
                ic  = SEGS.get(row["Segment"],{}).get("icon","")
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:0.5rem;">
                    <div style="font-size:0.78rem;color:{SUBTEXT};width:110px;">{ic} {row['Segment']}</div>
                    <div style="flex:1;height:6px;background:{BORDER};border-radius:10px;overflow:hidden;">
                        <div style="height:100%;width:{pct}%;background:{bc};border-radius:10px;"></div>
                    </div>
                    <div style="font-size:0.75rem;color:{SUBTEXT};font-weight:600;width:32px;text-align:right;">{pct}%</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:{EMPTY_BG};border:1.5px dashed {BORDER};border-radius:12px;
                 padding:4rem 2rem;text-align:center;">
                <div style="font-size:2rem;margin-bottom:0.5rem;">👥</div>
                <div style="color:{SUBTEXT};font-size:0.9rem;font-weight:500;">Enter values and click Predict</div>
                <div style="color:{MUTED};font-size:0.78rem;margin-top:0.5rem;line-height:1.6;">
                    The model will classify the customer as<br>
                    <b>High-Value · Regular · Occasional · At-Risk</b>
                </div>
            </div>""", unsafe_allow_html=True)


# ── TAB 3: How to Use ────────────────────────────────────────────────────
with tab3:
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div style="background:{SURFACE};border:1px solid {BORDER};border-radius:12px;padding:1.5rem;">
            <div style="font-size:1.5rem;margin-bottom:0.5rem;">🎯</div>
            <div style="font-size:0.95rem;font-weight:600;color:{TEXT};margin-bottom:0.8rem;">Product Recommendations</div>
            <div style="color:{SUBTEXT};font-size:0.82rem;line-height:1.7;">
                1. Go to the <b>Product Recommendations</b> tab<br>
                2. Type any product name or keyword<br>
                3. Click <b>Search →</b><br>
                4. See 5 products customers bought together<br><br>
                <span style="color:{MUTED};font-size:0.75rem;">
                Try: WHITE HANGING HEART · REGENCY CAKESTAND · CERAMIC JAR
                </span>
            </div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div style="background:{SURFACE};border:1px solid {BORDER};border-radius:12px;padding:1.5rem;">
            <div style="font-size:1.5rem;margin-bottom:0.5rem;">👥</div>
            <div style="font-size:0.95rem;font-weight:600;color:{TEXT};margin-bottom:0.8rem;">Customer Segmentation</div>
            <div style="color:{SUBTEXT};font-size:0.82rem;line-height:1.7;">
                1. Go to <b>Customer Segmentation</b> tab<br>
                2. Enter <b>Recency</b> — days since last purchase<br>
                3. Enter <b>Frequency</b> — number of orders<br>
                4. Enter <b>Monetary</b> — total spend in £<br>
                5. Click <b>⚡ Predict Segment</b>
            </div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div style="background:{SURFACE};border:1px solid {BORDER};border-radius:12px;padding:1.5rem;">
            <div style="font-size:1.5rem;margin-bottom:0.5rem;">📊</div>
            <div style="font-size:0.95rem;font-weight:600;color:{TEXT};margin-bottom:0.8rem;">RFM & Segments Explained</div>
            <div style="color:{SUBTEXT};font-size:0.82rem;line-height:1.7;">
                <b>R</b>ecency — lower = bought recently ✅<br>
                <b>F</b>requency — higher = orders more ✅<br>
                <b>M</b>onetary — higher = spends more ✅<br><br>
                👑 <b>High-Value</b> — VIP customers<br>
                ⭐ <b>Regular</b> — Loyal buyers<br>
                🔄 <b>Occasional</b> — Infrequent buyers<br>
                ⚠️ <b>At-Risk</b> — Need win-back
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="text-align:center;color:{MUTED};font-size:0.75rem;margin-top:2rem;padding-top:1rem;
         border-top:1px solid {BORDER};">
        Shopper Spectrum &nbsp;·&nbsp; Random Forest (93% accuracy) &nbsp;·&nbsp;
        Item-Based Collaborative Filtering &nbsp;·&nbsp; Built by Naman · Manipal University Jaipur
    </div>""", unsafe_allow_html=True)

