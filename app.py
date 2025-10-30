# ✅ AI E-Commerce Recommender UI
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from recommend import (
    rating_matrix,
    products,
    item_similarity,
    user_similarity,
    hybrid_recommend,
    get_hybrid_scores,
)

# -------------------- Page Settings --------------------
st.set_page_config(page_title="AI E-Commerce Recommender",
                   layout="wide", page_icon="🛒")

# Ensure output folder exists
os.makedirs("output", exist_ok=True)

# -------------------- Theme Toggle --------------------
theme = st.sidebar.radio("Theme", ["Light", "Dark"], key="theme_toggle")
if theme == "Dark":
    st.markdown("""
    <style>
    body, .stApp { background-color: #121212 !important; color: #e0e0e0 !important; }
    .stButton>button { background-color: #2563eb !important; color: white !important; }
    </style>
    """, unsafe_allow_html=True)

# -------------------- Title --------------------
st.markdown("<h1 style='text-align:center;'>🛍️ AI-Powered Product Recommendation System</h1>", unsafe_allow_html=True)
st.write("---")

# -------------------- Sidebar Controls --------------------
alpha = st.sidebar.slider("Hybrid Weight", 0.0, 1.0, 0.6)
top_n = st.sidebar.slider("Top N Products", 1, 10, 5)

price_limit = st.sidebar.number_input("Max Price (₹) Optional", min_value=0.0, value=0.0)
price_limit = None if price_limit == 0 else price_limit

cats = ["All"] + sorted(products["category"].unique())
cat = st.sidebar.selectbox("Category Filter", cats)
cat = None if cat == "All" else cat

save = st.sidebar.checkbox("Save Graph Images", value=True)

# -------------------- User Selection --------------------
user = st.selectbox("Select User", rating_matrix.index)

# Maintain state
if "recs" not in st.session_state:
    st.session_state.recs = None
    st.session_state.scores = None

# -------------------- Generate Recommendations --------------------
if st.button("✨ Recommend"):
    st.session_state.recs = hybrid_recommend(user, alpha, top_n, price_limit, cat)
    st.session_state.scores = get_hybrid_scores(user, alpha)

if st.session_state.recs is None or st.session_state.recs.empty:
    st.info("👉 Select a user & click Recommend to see results.")
    st.stop()

# -------------------- Show Recommendations --------------------
st.subheader(f"✅ Recommended Products for {user}")
for _, row in st.session_state.recs.iterrows():
    col1, col2 = st.columns([1, 4])

    img = str(row["image"]).strip()
    if not img.lower().startswith("images/"):
        img = os.path.join("images", img)

    with col1:
        if os.path.exists(img):
            st.image(img, width=110)
        else:
            st.markdown("📦")

    with col2:
        st.markdown(f"### {row['product_name']}")
        stars = "⭐" * min(5, max(1, int(row['score'])))
        st.write(f"Predicted Score: {row['score']:.2f} {stars}")
        st.write(f"Price: ₹{row['price']}")
        st.write(f"Category: {row['category']}")

    st.markdown("---")

# -------------------- Graph Selection --------------------
st.subheader("📊 Recommendation Insights")
opt = st.radio(
    "Select Visualization",
    ["Hybrid Score Chart", "Item Similarity Heatmap", "User Similarity Heatmap", "Similar Users Table"],
    key="vis_radio"
)

already = rating_matrix.loc[user] > 0
scores = st.session_state.scores[~already].sort_values(ascending=False).head(20)

# -------------------- Graphs --------------------
if opt == "Hybrid Score Chart":
    fig, ax = plt.subplots(figsize=(8,4))
    scores.plot(kind="bar", ax=ax)
    ax.set_title("Top Predicted Recommendation Scores")
    st.pyplot(fig)
    if save: fig.savefig(f"output/scores_{user}.png")

    

elif opt == "Item Similarity Heatmap":
    top_items = (rating_matrix > 0).sum().sort_values(ascending=False).head(20).index
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(item_similarity.loc[top_items, top_items], cmap="coolwarm", ax=ax, linecolor='black',   # color of the cell borders
    linewidths=0.5)
    ax.set_title("Item Correlation Heatmap")
    st.pyplot(fig)
    if save: fig.savefig(f"output/item_sim_{user}.png")

elif opt == "User Similarity Heatmap":
    sim_users = user_similarity[user].sort_values(ascending=False).head(8).index
    fig, ax = plt.subplots(figsize=(7,5))
    sns.heatmap(user_similarity.loc[sim_users, sim_users], cmap="Blues", ax=ax ,linecolor='black',   # color of the cell borders
    linewidths=0.5)
    ax.set_title("User Similarity Heatmap")
    st.pyplot(fig)
    if save: fig.savefig(f"output/user_sim_{user}.png")

elif opt == "Similar Users Table":
    sim = user_similarity[user].sort_values(ascending=False).drop(user)
    st.table(pd.DataFrame({"User": sim.index, "Similarity Score": sim.values}).head(10))
