"""
Hybrid recommender module for AI E-Commerce.
Exports:
 - rating_matrix : pandas DataFrame (users x products)
 - products : product metadata DataFrame
 - item_similarity : item-item cosine similarity DataFrame
 - user_similarity : user-user cosine similarity DataFrame
 - hybrid_recommend(...) -> pandas DataFrame of recommendations
 - get_hybrid_scores(...) -> pandas Series of hybrid scores
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

# -------------------- Load Data --------------------
RATINGS_CSV = "data/ecommerce_ratings.csv"
PRODUCTS_CSV = "data/ecommerce_products.csv"

ratings = pd.read_csv(RATINGS_CSV)
products = pd.read_csv(PRODUCTS_CSV)

# Ensure optional columns exist
if "price" not in products.columns:
    products["price"] = np.nan
if "category" not in products.columns:
    products["category"] = "General"
if "image" not in products.columns:
    products["image"] = ""

# -------------------- Build Rating Matrix (NaN-Safe) --------------------
rating_matrix = ratings.pivot_table(
    index="user",
    columns="product",
    values="rating"
)

# Include all products from the dataset
rating_matrix = rating_matrix.reindex(columns=products["product"], fill_value=0)

# Fill any remaining NaNs with 0
rating_matrix = rating_matrix.fillna(0)

# -------------------- Compute Similarity Matrices --------------------
item_similarity = pd.DataFrame(
    cosine_similarity(rating_matrix.T),
    index=rating_matrix.columns,
    columns=rating_matrix.columns
)

user_similarity = pd.DataFrame(
    cosine_similarity(rating_matrix),
    index=rating_matrix.index,
    columns=rating_matrix.index
)

# -------------------- Score Functions --------------------
def item_based_score(user_id):
    """Predict score for each item based on item similarity & user's ratings."""
    user_ratings = rating_matrix.loc[user_id]
    scores = item_similarity.dot(user_ratings) / (item_similarity.sum(axis=1) + 1e-9)
    return scores

def user_based_score(user_id):
    """Predict score for each item based on similar users' ratings."""
    sims = user_similarity[user_id].copy()
    sims = sims.drop(user_id)  # exclude self
    other_ratings = rating_matrix.loc[sims.index]
    weighted = sims.values.dot(other_ratings) / (sims.values.sum() + 1e-9)
    scores = pd.Series(weighted, index=rating_matrix.columns)
    return scores

def get_hybrid_scores(user_id, alpha=0.6):
    """
    Compute hybrid score = alpha * user_based + (1-alpha) * item_based
    Returns a pandas Series indexed by product id.
    """
    if user_id not in rating_matrix.index:
        raise ValueError(f"User {user_id} not found in rating matrix.")
    ib = item_based_score(user_id)
    ub = user_based_score(user_id)
    hybrid = alpha * ub + (1 - alpha) * ib
    return hybrid

# -------------------- Hybrid Recommendation --------------------
def hybrid_recommend(user_id, alpha=0.6, top_n=5, price_limit=None, category=None):
    """
    Return top_n recommended products for user_id.
    Returns DataFrame: product, score, product_name, category, price, image
    Filters:
      - price_limit : float (keep price <= price_limit)
      - category : str (filter category)
    """
    hybrid = get_hybrid_scores(user_id, alpha=alpha)

    # exclude already rated items
    already_rated = rating_matrix.loc[user_id] > 0
    candidates = hybrid[~already_rated]

    # merge with product metadata
    recs = candidates.sort_values(ascending=False).head(top_n).reset_index()
    recs.columns = ["product", "score"]
    recs = recs.merge(products, on="product", how="left")

    # apply filters
    if price_limit is not None and "price" in recs.columns:
        recs = recs[recs["price"].notna() & (recs["price"] <= price_limit)]
    if category is not None and "category" in recs.columns:
        recs = recs[recs["category"] == category]

    return recs.reset_index(drop=True)
