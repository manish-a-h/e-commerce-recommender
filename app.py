import streamlit as st
import pandas as pd
import pickle
from recommender.recommend import load_data, load_model, get_item_recommendations, get_user_recommendations
from sklearn.neighbors import NearestNeighbors

# Set the title
st.set_page_config(page_title="E-commerce Recommender", layout="centered")
st.title("🛍️ E-commerce Recommender System")

# Load data and models
@st.cache_resource
def load_all():
    df, user_item_matrix, item_id_to_index, index_to_item_id, user_id_to_index = load_data()
    
    item_model = load_model("models/item_similarity.pkl")
    
    with open("models/user_similarity.pkl", "rb") as f:
        user_model = pickle.load(f)

    return df, user_item_matrix, item_id_to_index, index_to_item_id, user_id_to_index, item_model, user_model

df, user_item_matrix, item_id_to_index, index_to_item_id, user_id_to_index, item_model, user_model = load_all()

# UI Tabs
tab1, tab2 = st.tabs(["🎯 Item-Based", "👤 User-Based"])

with tab1:
    st.subheader("Item-Based Recommendations")

    item_id_input = st.text_input("Enter Item ID:")
    if st.button("Get Similar Items"):
        try:
            item_id = int(item_id_input)
            if item_id in item_id_to_index:
                recommendations = get_item_recommendations(
                    item_id, item_model, user_item_matrix, item_id_to_index, index_to_item_id
                )
                st.success(f"Top recommendations for Item ID {item_id}:")
                st.write(recommendations)
            else:
                st.warning("Item ID not found in dataset.")
        except ValueError:
            st.error("Please enter a valid integer for Item ID.")

with tab2:
    st.subheader("User-Based Recommendations")

    user_id_input = st.text_input("Enter User ID:")
    if st.button("Get Recommended Items"):
        try:
            user_id = int(user_id_input)
            if user_id in user_id_to_index:
                recommendations = get_user_recommendations(
                    user_id, user_model, user_item_matrix, user_id_to_index, index_to_item_id
                )
                st.success(f"Top recommendations for User ID {user_id}:")
                st.write(recommendations)
            else:
                st.warning("User ID not found in dataset.")
        except ValueError:
            st.error("Please enter a valid integer for User ID.")
