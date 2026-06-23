"""
train_model.py
================
Shopper Spectrum -- Model Training Script

Standalone script (run outside the notebook) that:
  1. Loads and cleans the online_retail.csv transaction data
  2. Builds customer-level RFM (Recency, Frequency, Monetary) features
  3. Labels each customer's segment using a rule-based RFM quartile-scoring
     system (business logic, NOT machine learning)
  4. Trains a SUPERVISED classifier (Random Forest) to predict a customer's
     segment from their RFM values -- this is the model the Streamlit app uses
  5. Builds an item-based collaborative filtering recommendation engine
     (cosine similarity over the customer-product purchase matrix)
  6. Saves every artifact needed by app.py into the saved_models/ folder

Run this once before launching the Streamlit app:
    python train_model.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = "online_retail.csv"          
SAVE_DIR = "saved_models"
RANDOM_STATE = 42

ADMIN_CODES = ["POST", "M", "C2", "DOT", "BANK CHARGES", "PADS", "CRUK"]


def load_and_clean_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)

    df = df.drop_duplicates()
    df = df.dropna(subset=["CustomerID"])
    df = df.dropna(subset=["Description"])
    df = df[~df["InvoiceNo"].str.startswith("C")]          
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]   

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    df["CustomerID"] = df["CustomerID"].astype(int)

    return df.reset_index(drop=True)


def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
    ).reset_index()

    rfm["Recency_log"] = np.log1p(rfm["Recency"])
    rfm["Frequency_log"] = np.log1p(rfm["Frequency"])
    rfm["Monetary_log"] = np.log1p(rfm["Monetary"])

    return rfm


def label_segments(rfm: pd.DataFrame) -> pd.DataFrame:
    rfm = rfm.copy()

    # Recency: lower days-since-last-purchase is better -> score 4 (best) to 1 (worst)
    rfm["R_score"] = pd.qcut(rfm["Recency"], 4, labels=[4, 3, 2, 1]).astype(int)
    # Frequency / Monetary: higher is better -> score 1 (worst) to 4 (best)
    rfm["F_score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    rfm["M_score"] = pd.qcut(rfm["Monetary"], 4, labels=[1, 2, 3, 4]).astype(int)

    rfm["RFM_Score"] = rfm["R_score"] + rfm["F_score"] + rfm["M_score"]

    def _label(score):
        if score >= 10:
            return "High-Value"
        elif score >= 8:
            return "Regular"
        elif score >= 5:
            return "Occasional"
        else:
            return "At-Risk"

    rfm["Segment"] = rfm["RFM_Score"].apply(_label)
    return rfm

def train_segment_classifier(rfm: pd.DataFrame):
    features = ["Recency_log", "Frequency_log", "Monetary_log"]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(rfm["Segment"])

    scaler = StandardScaler()
    X = scaler.fit_transform(rfm[features])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Hyperparameter tuning via GridSearchCV (this IS the supervised ML step)
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [6, 8, 10],
        "min_samples_split": [2, 5],
    }
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
        param_grid,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    model = grid_search.best_estimator_
    pred = model.predict(X_test)

    print("Best hyperparameters:", grid_search.best_params_)
    print("Best CV F1 (macro)  :", round(grid_search.best_score_, 4))
    print("Test Accuracy       :", round(accuracy_score(y_test, pred), 4))
    print("Test F1 (macro)     :", round(f1_score(y_test, pred, average="macro"), 4))
    print()
    print(classification_report(y_test, pred, target_names=label_encoder.classes_))

    return model, scaler, label_encoder, features


def build_recommender(df: pd.DataFrame):
    df_products = df[~df["StockCode"].str.upper().isin(ADMIN_CODES)].copy()

    desc_lookup = df_products.groupby("StockCode")["Description"].agg(
        lambda x: x.value_counts().index[0]
    )

    # Binary purchase-indicator weighting (found to outperform raw-quantity weighting)
    basket = df_products.pivot_table(
        index="CustomerID", columns="StockCode", values="Quantity", aggfunc="sum", fill_value=0
    )
    basket_bin = (basket > 0).astype(int)

    item_sim = cosine_similarity(basket_bin.T)
    item_sim_df = pd.DataFrame(item_sim, index=basket_bin.columns, columns=basket_bin.columns)

    return item_sim_df, desc_lookup


def main():
    print("Loading & cleaning data...")
    df = load_and_clean_data(DATA_PATH)
    print(f"  Clean transactions: {df.shape[0]:,} rows, {df['CustomerID'].nunique():,} customers")

    print("\nBuilding RFM features...")
    rfm = build_rfm(df)

    print("\nLabeling segments (rule-based RFM quartile scoring)...")
    rfm = label_segments(rfm)
    print(rfm["Segment"].value_counts())

    print("\nTraining supervised segment classifier (Random Forest)...")
    model, scaler, label_encoder, features = train_segment_classifier(rfm)

    print("\nBuilding product recommendation engine (item-based collaborative filtering)...")
    item_sim_df, desc_lookup = build_recommender(df)
    print(f"  Similarity matrix shape: {item_sim_df.shape}")

    print("\nSaving artifacts...")
    os.makedirs(SAVE_DIR, exist_ok=True)
    joblib.dump(scaler, f"{SAVE_DIR}/rfm_scaler.pkl")
    joblib.dump(model, f"{SAVE_DIR}/segment_classifier.pkl")
    joblib.dump(label_encoder, f"{SAVE_DIR}/segment_label_encoder.pkl")
    joblib.dump(item_sim_df, f"{SAVE_DIR}/item_similarity_matrix.pkl")
    joblib.dump(desc_lookup, f"{SAVE_DIR}/product_description_lookup.pkl")

    for f in sorted(os.listdir(SAVE_DIR)):
        print(" -", f)

    print("\nDone. You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()
