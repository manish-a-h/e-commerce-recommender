# Save this in a separate script (e.g., train_model.py)
import pickle
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import pandas as pd

df = pd.read_csv("data/processed/interactions.csv")
user_ids = df["visitorid"].astype("category").cat.codes
item_ids = df["itemid"].astype("category").cat.codes
df["user_index"] = user_ids
df["item_index"] = item_ids

user_item_matrix = csr_matrix((df["rating"], (df["user_index"], df["item_index"])))

item_user_matrix = user_item_matrix.T  # items as rows

model = NearestNeighbors(metric="cosine", algorithm="brute")
model.fit(item_user_matrix)

with open("models/item_similarity.pkl", "wb") as f:
    pickle.dump(model, f)
