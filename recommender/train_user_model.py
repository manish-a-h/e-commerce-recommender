# train_user_model.py

import pandas as pd
import pickle
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import os

def load_data():
    interactions_path = "data/processed/interactions.csv"
    df = pd.read_csv(interactions_path)

    # Convert visitorid and itemid to categorical index
    df["user_index"] = df["visitorid"].astype("category").cat.codes
    df["item_index"] = df["itemid"].astype("category").cat.codes

    user_item_matrix = csr_matrix(
        (df["rating"], (df["user_index"], df["item_index"]))
    )

    return df, user_item_matrix

def train_and_save_user_similarity_model():
    df, user_item_matrix = load_data()

    print("✅ Training user-based similarity model...")
    model = NearestNeighbors(metric="cosine", algorithm="brute")
    model.fit(user_item_matrix)

    os.makedirs("models", exist_ok=True)
    with open("models/user_similarity.pkl", "wb") as f:
        pickle.dump(model, f)

    print("✅ Model saved to models/user_similarity.pkl")

if __name__ == "__main__":
    train_and_save_user_similarity_model()
