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
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif; }

/* Background */
.stApp { background: #0a0a0f; }

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem; max-width: 1400px; }

/* Hero Section */
.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 24px;
    padding: 3rem 3.5rem;
    margin-bottom: 2rem;
    border: 1px solid rgba(99,102,241,0.3);
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-badge {
    display: inline-block;
    background: rgba(99,102,241,0.2);
    border: 1px solid rgba(99,102,241,0.5);
    color: #818cf8;
    padding: 0.3rem 1rem;
    border-radius: 50px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero h1 {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #ffffff 0%, #a5b4fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0.5rem 0;
    line-height: 1.2;
}
.hero p {
    color: #94a3b8;
    font-size: 1.05rem;
    max-width: 600px;
    margin-top: 0.5rem;
    line-height: 1.6;
}
.hero-stats {
    display: flex;
    gap: 2.5rem;
    margin-top: 2rem;
}
.stat-item { text-align: left; }
.stat-number {
    font-size: 1.8rem;
    font-weight: 700;
    color: #a5b4fc;
}
.stat-label {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: #111827;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #1f2937;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #6b7280;
    border-radius: 8px;
    font-weight: 500;
    padding: 0.6rem 1.5rem;
    font-size: 0.9rem;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important;
}

/* Input fields */
.stTextInput input, .stNumberInput input {
    background: #111827 !important;
    border: 1px solid #1f2937 !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
    padding: 0.7rem 1rem !important;
    font-size: 0.9rem !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: #4f46e5 !important;
    box-shadow: 0 0 0 3px rgba(79,70,229,0.15) !important;
}

/* Buttons */
.stButton button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 2rem !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.3px !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
.stButton button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(79,70,229,0.4) !important;
}

/* Product cards */
.product-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 16px;
    padding: 1.3rem;
    height: 180px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.product-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #4f46e5, #7c3aed);
    border-radius: 16px 16px 0 0;
}
.product-card:hover {
    border-color: #4f46e5;
    transform: translateY(-4px);
    box-shadow: 0 12px 35px rgba(79,70,229,0.2);
}
.product-rank {
    font-size: 0.7rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
}
.product-name {
    font-size: 0.88rem;
    color: #e2e8f0;
    font-weight: 500;
    line-height: 1.4;
    flex: 1;
    margin: 0.5rem 0;
}
.product-code {
    font-size: 0.72rem;
    color: #4b5563;
    font-family: monospace;
}
.similarity-bar {
    background: #1f2937;
    border-radius: 50px;
    height: 4px;
    margin-top: 0.5rem;
    overflow: hidden;
}
.similarity-fill {
    height: 100%;
    background: linear-gradient(90deg, #4f46e5, #7c3aed);
    border-radius: 50px;
}

/* Segment cards */
.segment-result {
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    margin: 1rem 0;
    position: relative;
    overflow: hidden;
}
.segment-icon { font-size: 3.5rem; margin-bottom: 0.5rem; }
.segment-name {
    font-size: 2rem;
    font-weight: 800;
    margin: 0.3rem 0;
}
.segment-desc {
    font-size: 0.95rem;
    opacity: 0.85;
    max-width: 500px;
    margin: 0 auto;
    line-height: 1.6;
}

/* RFM input cards */
.rfm-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.5rem;
}
.rfm-label {
    font-size: 0.75rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
    margin-bottom: 0.3rem;
}

/* Confidence bars */
.conf-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin: 0.6rem 0;
}
.conf-label { color: #94a3b8; font-size: 0.85rem; width: 110px; }
.conf-bar-bg {
    flex: 1;
    background: #1f2937;
    border-radius: 50px;
    height: 8px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 50px;
    background: linear-gradient(90deg, #4f46e5, #7c3aed);
}
.conf-pct { color: #a5b4fc; font-size: 0.82rem; font-weight: 600; width: 40px; text-align: right; }

/* Section headers */
.section-header {
    margin: 1.5rem 0 1rem 0;
}
.section-header h3 {
    color: #f1f5f9;
    font-size: 1.1rem;
    font-weight: 600;
    margin: 0;
}
.section-header p {
    color: #64748b;
    font-size: 0.85rem;
    margin: 0.2rem 0 0 0;
}

/* Search result label */
.matched-label {
    background: rgba(79,70,229,0.15);
    border: 1px solid rgba(79,70,229,0.3);
    border-radius: 10px;
    padding: 0.8rem 1.2rem;
    color: #a5b4fc;
    font-size: 0.9rem;
    margin-bottom: 1.5rem;
}
.matched-label span { font-weight: 700; color: #818cf8; }

/* Divider */
.custom-divider {
    border: none;
    border-top: 1px solid #1f2937;
    margin: 2rem 0;
}

/* Info box */
.info-box {
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 12px;
    padding: 1rem 1.3rem;
    color: #94a3b8;
    font-size: 0.85rem;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

def build_similarity_matrix():
    df = pd.read_csv(DATA_PATH)
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df = df.drop_duplicates()
    df = df.dropna(subset=["CustomerID"])
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df = df[~df["InvoiceNo"].str.startswith("C")]
    df_products = df[~df["StockCode"].str.upper().isin(ADMIN_CODES)]
    basket = df_products.pivot_table(
        index="CustomerID", columns="StockCode",
        values="Quantity", aggfunc="sum", fill_value=0
    )
    basket_bin = (basket > 0).astype(int)
    sim = cosine_similarity(basket_bin.T)
    item_sim_df = pd.DataFrame(sim, index=basket_bin.columns, columns=basket_bin.columns)
    joblib.dump(item_sim_df, os.path.join(SAVE_DIR, "item_similarity_matrix.pkl"))
    return item_sim_df


@st.cache_resource
def load_artifacts():
    required = ["rfm_scaler.pkl", "segment_classifier.pkl",
                "segment_label_encoder.pkl", "product_description_lookup.pkl"]
    if any(not os.path.exists(os.path.join(SAVE_DIR, f)) for f in required):
        return None

    scaler        = joblib.load(os.path.join(SAVE_DIR, "rfm_scaler.pkl"))
    model         = joblib.load(os.path.join(SAVE_DIR, "segment_classifier.pkl"))
    label_encoder = joblib.load(os.path.join(SAVE_DIR, "segment_label_encoder.pkl"))
    desc_lookup   = joblib.load(os.path.join(SAVE_DIR, "product_description_lookup.pkl"))

    sim_path = os.path.join(SAVE_DIR, "item_similarity_matrix.pkl")
    if os.path.exists(sim_path):
        item_sim_df = joblib.load(sim_path)
    else:
        with st.spinner("⚙️ Building recommendation engine (first time only, ~30 sec)..."):
            item_sim_df = build_similarity_matrix()

    return {"scaler": scaler, "model": model, "label_encoder": label_encoder,
            "item_sim_df": item_sim_df, "desc_lookup": desc_lookup}


SEGMENT_CONFIG = {
    "High-Value": {
        "icon": "👑",
        "color": "#f59e0b",
        "bg": "linear-gradient(135deg, #1c1408 0%, #2d1f07 100%)",
        "border": "#f59e0b",
        "desc": "Your most valuable customers — recent, frequent, and high-spending. Treat them like royalty with exclusive offers and VIP perks.",
        "action": "💎 Recommended Action: Launch VIP loyalty program & early access offers"
    },
    "Regular": {
        "icon": "⭐",
        "color": "#3b82f6",
        "bg": "linear-gradient(135deg, #071428 0%, #0c1f3d 100%)",
        "border": "#3b82f6",
        "desc": "Solid, dependable customers with consistent purchase behavior. Great candidates for upselling and loyalty rewards.",
        "action": "🎯 Recommended Action: Introduce loyalty points & bundle offers"
    },
    "Occasional": {
        "icon": "🔄",
        "color": "#f97316",
        "bg": "linear-gradient(135deg, #1a0e05 0%, #2a1608 100%)",
        "border": "#f97316",
        "desc": "Infrequent buyers who need a nudge. Targeted campaigns can convert them into regular customers.",
        "action": "📧 Recommended Action: Personalized re-engagement email campaigns"
    },
    "At-Risk": {
        "icon": "⚠️",
        "color": "#ef4444",
        "bg": "linear-gradient(135deg, #1a0505 0%, #2d0808 100%)",
        "border": "#ef4444",
        "desc": "Haven't purchased in a long time. Act now before they're lost — win-back campaigns are critical here.",
        "action": "🚨 Recommended Action: Urgent win-back offer with heavy discount"
    },
}


def recommend(product_name, artifacts, n=5):
    matches = artifacts["desc_lookup"][
        artifacts["desc_lookup"].str.contains(product_name, case=False, na=False, regex=False)
    ]
    if len(matches) == 0:
        return None, None
    code = matches.index[0]
    sims = artifacts["item_sim_df"][code].drop(code).sort_values(ascending=False).head(n)
    results = pd.DataFrame({
        "StockCode": sims.index,
        "Product": artifacts["desc_lookup"].loc[sims.index].values,
        "Score": sims.values,
    })
    return artifacts["desc_lookup"].loc[code], results


def predict(recency, frequency, monetary, artifacts):
    feat = pd.DataFrame({
        "Recency_log":   [np.log1p(recency)],
        "Frequency_log": [np.log1p(frequency)],
        "Monetary_log":  [np.log1p(monetary)],
    })
    scaled = artifacts["scaler"].transform(feat)
    enc    = artifacts["model"].predict(scaled)[0]
    proba  = artifacts["model"].predict_proba(scaled)[0]
    segment = artifacts["label_encoder"].inverse_transform([enc])[0]
    proba_df = pd.DataFrame({
        "Segment": artifacts["label_encoder"].classes_,
        "Prob": proba,
    }).sort_values("Prob", ascending=False).reset_index(drop=True)
    return segment, proba_df

def main():
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">🛒 ML-Powered Analytics</div>
        <h1>Shopper Spectrum</h1>
        <p>Customer segmentation & product recommendations powered by Random Forest classification and collaborative filtering.</p>
        <div class="hero-stats">
            <div class="stat-item">
                <div class="stat-number">4,338</div>
                <div class="stat-label">Customers Analyzed</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">93%</div>
                <div class="stat-label">Model Accuracy</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">3,659</div>
                <div class="stat-label">Products Indexed</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">4</div>
                <div class="stat-label">Customer Segments</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    artifacts = load_artifacts()
    if artifacts is None:
        st.error("⚠️ Model files not found. Please run `python train_model.py` first.")
        st.stop()

    tab1, tab2 = st.tabs(["🎯  Product Recommendations", "👥  Customer Segmentation"])

    with tab1:
        st.markdown("""
        <div class="section-header">
            <h3>Product Recommendation Engine</h3>
            <p>Based on real purchase co-occurrence across 4,338 customers — finds what people actually buy together</p>
        </div>
        """, unsafe_allow_html=True)

        col_input, col_btn = st.columns([4, 1])
        with col_input:
            query = st.text_input("", placeholder="🔍  Type a product name, e.g.  WHITE HANGING HEART  or  REGENCY CAKESTAND", label_visibility="collapsed")
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            search = st.button("Find Similar →")

        st.markdown("""
        <div class="info-box">
            💡 <b>How it works:</b> Enter any product name (or part of it). The engine finds the 5 most similar products 
            based on which customers bought them together — no text matching, pure behavioral similarity.
        </div>
        """, unsafe_allow_html=True)

        if search:
            if not query.strip():
                st.warning("Please enter a product name.")
            else:
                matched_name, results = recommend(query, artifacts)
                if results is None:
                    st.error(f"No product found matching **'{query}'**. Try a shorter keyword.")
                else:
                    st.markdown(f"""
                    <div class="matched-label">
                        ✅ Showing recommendations for: <span>{matched_name}</span>
                    </div>
                    """, unsafe_allow_html=True)

                    cols = st.columns(5)
                    for i, (_, row) in enumerate(results.iterrows()):
                        pct = int(row['Score'] * 100)
                        with cols[i]:
                            st.markdown(f"""
                            <div class="product-card">
                                <div class="product-rank">#{i+1} Match</div>
                                <div class="product-name">{row['Product']}</div>
                                <div>
                                    <div class="product-code">{row['StockCode']}</div>
                                    <div class="similarity-bar">
                                        <div class="similarity-fill" style="width:{pct}%"></div>
                                    </div>
                                    <div style="color:#6b7280; font-size:0.72rem; margin-top:4px;">{row['Score']:.3f} similarity</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

    with tab2:
        st.markdown("""
        <div class="section-header">
            <h3>Customer Segment Predictor</h3>
            <p>Enter a customer's RFM values to instantly predict their segment using our trained Random Forest model</p>
        </div>
        """, unsafe_allow_html=True)

        col_form, col_result = st.columns([1, 1], gap="large")

        with col_form:
            st.markdown("""
            <div style="background:#111827; border:1px solid #1f2937; border-radius:16px; padding:1.5rem; margin-bottom:1rem;">
                <div style="color:#94a3b8; font-size:0.8rem; text-transform:uppercase; letter-spacing:1px; font-weight:600; margin-bottom:1rem;">RFM Input Parameters</div>
            """, unsafe_allow_html=True)

            recency   = st.number_input("📅  Recency — Days since last purchase", min_value=0, max_value=1000, value=30, step=1)
            frequency = st.number_input("🔁  Frequency — Number of purchases made", min_value=1, max_value=500, value=5, step=1)
            monetary  = st.number_input("💷  Monetary — Total spend in £", min_value=0.0, max_value=500000.0, value=500.0, step=10.0)

            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("""
            <div class="info-box" style="margin-bottom:1rem;">
                <b>RFM Guide:</b><br>
                📅 <b>Recency:</b> Lower = better (bought recently)<br>
                🔁 <b>Frequency:</b> Higher = better (buys often)<br>
                💷 <b>Monetary:</b> Higher = better (spends more)
            </div>
            """, unsafe_allow_html=True)

            predict_btn = st.button("⚡  Predict Segment")

        with col_result:
            if predict_btn:
                segment, proba_df = predict(recency, frequency, monetary, artifacts)
                cfg = SEGMENT_CONFIG[segment]

                st.markdown(f"""
                <div class="segment-result" style="background:{cfg['bg']}; border:2px solid {cfg['border']}40;">
                    <div class="segment-icon">{cfg['icon']}</div>
                    <div class="segment-name" style="color:{cfg['color']};">{segment}</div>
                    <div class="segment-desc" style="color:#94a3b8;">{cfg['desc']}</div>
                    <div style="margin-top:1.2rem; background:rgba(255,255,255,0.05); border-radius:10px; padding:0.7rem 1rem; font-size:0.82rem; color:{cfg['color']};">
                        {cfg['action']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                <div style="color:#94a3b8; font-size:0.8rem; text-transform:uppercase; letter-spacing:1px; font-weight:600; margin:1.2rem 0 0.7rem 0;">
                    Model Confidence
                </div>
                """, unsafe_allow_html=True)

                for _, row in proba_df.iterrows():
                    pct = int(row['Prob'] * 100)
                    seg_cfg = SEGMENT_CONFIG.get(row['Segment'], {})
                    color = seg_cfg.get('color', '#4f46e5')
                    st.markdown(f"""
                    <div class="conf-row">
                        <div class="conf-label">{seg_cfg.get('icon','')} {row['Segment']}</div>
                        <div class="conf-bar-bg">
                            <div class="conf-bar-fill" style="width:{pct}%; background:{color};"></div>
                        </div>
                        <div class="conf-pct">{pct}%</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="height:100%; display:flex; flex-direction:column; align-items:center; justify-content:center; 
                     background:#111827; border:1px dashed #1f2937; border-radius:20px; padding:3rem; text-align:center;">
                    <div style="font-size:3rem; margin-bottom:1rem;">🎯</div>
                    <div style="color:#4b5563; font-size:0.95rem;">Enter RFM values and click<br><b style="color:#6b7280;">Predict Segment</b></div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("""
    <hr class="custom-divider">
    <div style="text-align:center; color:#374151; font-size:0.78rem; padding-bottom:1rem;">
        Shopper Spectrum &nbsp;·&nbsp; Random Forest Classifier (93% accuracy) &nbsp;·&nbsp; Item-Based Collaborative Filtering &nbsp;·&nbsp; Built by Naman
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
