"""
app.py
=======
Shopper Spectrum -- Streamlit Web Application

Two modules:
  1. Product Recommendation : enter a product name, get 5 similar products
     (item-based collaborative filtering / cosine similarity)
  2. Customer Segmentation  : enter Recency, Frequency, Monetary, get the
     predicted customer segment (Random Forest classifier)

Before running this app, generate the model artifacts once with:
    python train_model.py

Then launch the app with:
    streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide",
)

SAVE_DIR = "saved_models"

SEGMENT_COLORS = {
    "High-Value": "#2e7d32",
    "Regular": "#1565c0",
    "Occasional": "#ef6c00",
    "At-Risk": "#c62828",
}

SEGMENT_DESCRIPTIONS = {
    "High-Value": "Recent, frequent, and high-spending customers. Your best customers -- prioritize retention and VIP treatment.",
    "Regular": "Solid, dependable repeat customers with above-average spend. Good candidates for loyalty programs.",
    "Occasional": "Infrequent buyers with moderate spend. Good targets for re-engagement campaigns and personalized offers.",
    "At-Risk": "Haven't purchased in a long time. Target with win-back campaigns before they're lost for good.",
}


# --------------------------------------------------------------------------
# Cached artifact loading
# --------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    missing = [
        f for f in [
            "rfm_scaler.pkl",
            "segment_classifier.pkl",
            "segment_label_encoder.pkl",
            "item_similarity_matrix.pkl",
            "product_description_lookup.pkl",
        ]
        if not os.path.exists(f"{SAVE_DIR}/{f}")
    ]
    if missing:
        return None

    scaler = joblib.load(f"{SAVE_DIR}/rfm_scaler.pkl")
    model = joblib.load(f"{SAVE_DIR}/segment_classifier.pkl")
    label_encoder = joblib.load(f"{SAVE_DIR}/segment_label_encoder.pkl")
    item_sim_df = joblib.load(f"{SAVE_DIR}/item_similarity_matrix.pkl")
    desc_lookup = joblib.load(f"{SAVE_DIR}/product_description_lookup.pkl")

    return {
        "scaler": scaler,
        "model": model,
        "label_encoder": label_encoder,
        "item_sim_df": item_sim_df,
        "desc_lookup": desc_lookup,
    }


def recommend_products(product_name: str, item_sim_df: pd.DataFrame, desc_lookup: pd.Series, n: int = 5):
    matches = desc_lookup[desc_lookup.str.contains(product_name, case=False, na=False, regex=False)]
    if len(matches) == 0:
        return None, None

    code = matches.index[0]
    matched_name = desc_lookup.loc[code]
    sims = item_sim_df[code].drop(code).sort_values(ascending=False).head(n)

    results = pd.DataFrame({
        "StockCode": sims.index,
        "Product": desc_lookup.loc[sims.index].values,
        "Similarity Score": sims.values.round(3),
    })
    return matched_name, results


def predict_segment(recency: float, frequency: float, monetary: float, artifacts: dict):
    features = pd.DataFrame({
        "Recency_log": [np.log1p(recency)],
        "Frequency_log": [np.log1p(frequency)],
        "Monetary_log": [np.log1p(monetary)],
    })
    scaled = artifacts["scaler"].transform(features)
    pred_encoded = artifacts["model"].predict(scaled)[0]
    pred_proba = artifacts["model"].predict_proba(scaled)[0]
    segment = artifacts["label_encoder"].inverse_transform([pred_encoded])[0]
    proba_df = pd.DataFrame({
        "Segment": artifacts["label_encoder"].classes_,
        "Probability": pred_proba,
    }).sort_values("Probability", ascending=False).reset_index(drop=True)
    return segment, proba_df


# --------------------------------------------------------------------------
# Main app
# --------------------------------------------------------------------------
def main():
    st.title("🛒 Shopper Spectrum")
    st.caption("Customer Segmentation & Product Recommendations for E-Commerce")

    artifacts = load_artifacts()

    if artifacts is None:
        st.error(
            "Model files not found in `saved_models/`. "
            "Please run `python train_model.py` first to generate them, "
            "then restart this app."
        )
        st.stop()

    tab1, tab2 = st.tabs(["🎯 Product Recommendation", "👥 Customer Segmentation"])

    # ---------------- Tab 1: Product Recommendation ----------------
    with tab1:
        st.subheader("Find similar products")
        st.write("Enter a product name (or part of it) to get 5 similar products customers also bought.")

        product_input = st.text_input("Product Name", placeholder="e.g. WHITE HANGING HEART T-LIGHT HOLDER")

        if st.button("Get Recommendations", type="primary"):
            if not product_input.strip():
                st.warning("Please enter a product name.")
            else:
                matched_name, results = recommend_products(
                    product_input, artifacts["item_sim_df"], artifacts["desc_lookup"]
                )
                if results is None:
                    st.warning(f"No product found matching '{product_input}'. Try a different keyword.")
                else:
                    st.success(f"Showing products similar to: **{matched_name}**")
                    cols = st.columns(5)
                    for i, (_, row) in enumerate(results.iterrows()):
                        with cols[i]:
                            st.markdown(
                                f"""
                                <div style="border:1px solid #ddd; border-radius:10px; padding:14px; min-height:160px;">
                                    <p style="font-size:13px; font-weight:600; line-height:1.3;">{row['Product']}</p>
                                    <p style="font-size:12px; color:gray;">Code: {row['StockCode']}</p>
                                    <p style="font-size:12px;">Similarity: <b>{row['Similarity Score']}</b></p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

    # ---------------- Tab 2: Customer Segmentation ----------------
    with tab2:
        st.subheader("Predict a customer's segment")
        st.write("Enter a customer's RFM values to predict which segment they belong to.")

        c1, c2, c3 = st.columns(3)
        with c1:
            recency = st.number_input("Recency (days since last purchase)", min_value=0, value=30, step=1)
        with c2:
            frequency = st.number_input("Frequency (number of purchases)", min_value=1, value=5, step=1)
        with c3:
            monetary = st.number_input("Monetary (total spend, £)", min_value=0.0, value=500.0, step=10.0)

        if st.button("Predict Cluster", type="primary"):
            segment, proba_df = predict_segment(recency, frequency, monetary, artifacts)
            color = SEGMENT_COLORS.get(segment, "#333333")

            st.markdown(
                f"""
                <div style="border-radius:10px; padding:20px; background-color:{color}20; border:2px solid {color};">
                    <h3 style="color:{color}; margin:0;">Predicted Segment: {segment}</h3>
                    <p style="margin-top:8px;">{SEGMENT_DESCRIPTIONS.get(segment, "")}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")
            st.write("**Prediction confidence by segment:**")
            st.bar_chart(proba_df.set_index("Segment"))

    st.divider()
    st.caption("Shopper Spectrum -- Built with a Random Forest customer segmentation model and an item-based collaborative filtering recommendation engine.")


if __name__ == "__main__":
    main()
