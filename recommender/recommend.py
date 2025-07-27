import pandas as pd
import pickle
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

def load_data():
    interactions_path = "data/processed/interactions.csv"
    df = pd.read_csv(interactions_path)

    # Convert visitorid and itemid to category codes
    user_ids = df["visitorid"].astype("category").cat.codes
    item_ids = df["itemid"].astype("category").cat.codes

    df["user_index"] = user_ids
    df["item_index"] = item_ids

    user_item_matrix = csr_matrix(
        (df["rating"], (df["user_index"], df["item_index"]))
    )

    # Create mappings
    item_id_to_index = dict(zip(df["itemid"], df["item_index"]))
    index_to_item_id = dict(zip(df["item_index"], df["itemid"]))
    user_id_to_index = dict(zip(df["visitorid"], df["user_index"]))

    return df, user_item_matrix, item_id_to_index, index_to_item_id, user_id_to_index


def load_model(model_path="models/item_similarity.pkl"):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def get_item_recommendations(item_id, model, user_item_matrix, item_id_to_index, index_to_item_id, n=5):
    if item_id not in item_id_to_index:
        return []
    
    item_index = item_id_to_index[item_id]
    item_vector = user_item_matrix.T[item_index].toarray().reshape(1, -1)

    distances, indices = model.kneighbors(item_vector, n_neighbors=n + 1)
    similar_item_indices = indices.flatten()[1:]  # Exclude the item itself

    recommended_item_ids = [index_to_item_id[i] for i in similar_item_indices]
    return recommended_item_ids


def get_user_recommendations(user_id, model, user_item_matrix, user_id_to_index, index_to_item_id, n=5):
    if user_id not in user_id_to_index:
        return []

    user_index = user_id_to_index[user_id]
    user_vector = user_item_matrix[user_index]

    distances, indices = model.kneighbors(user_vector, n_neighbors=n + 1)
    similar_users = indices.flatten()[1:]

    similar_users_items = user_item_matrix[similar_users].sum(axis=0)

    # Remove already interacted items
    user_items = user_item_matrix[user_index].nonzero()[1]
    for i in user_items:
        similar_users_items[0, i] = 0

    # Convert to dense array and sort
    top_items = similar_users_items.A1.argsort()[::-1][:n]
    recommended_item_ids = [index_to_item_id[i] for i in top_items]

    return recommended_item_ids


if __name__ == "__main__":
    # For command-line testing
    df, user_item_matrix, item_id_to_index, index_to_item_id, user_id_to_index = load_data()

    mode = input("Choose recommendation type (item/user): ").strip().lower()

    if mode == "item":
        model = load_model("models/item_similarity.pkl")
        item_id = int(input("Enter the item ID to get similar items: "))
        recommendations = get_item_recommendations(item_id, model, user_item_matrix, item_id_to_index, index_to_item_id)
        print(f"\n🔁 Recommended items similar to Item ID {item_id}:")
        print(recommendations)

    elif mode == "user":
        model = load_model("models/user_similarity.pkl")
        user_id = int(input("Enter the user ID to get recommended items: "))
        recommendations = get_user_recommendations(user_id, model, user_item_matrix, user_id_to_index, index_to_item_id)
        print(f"\n🧑 Recommended items for User ID {user_id}:")
        print(recommendations)

    else:
        print("Invalid mode. Choose either 'item' or 'user'.")
