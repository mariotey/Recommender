import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# %%
meta_df = pd.read_parquet("../data/output/meta_data.parquet")
user_item_itera = pd.read_parquet(
    "../data/output/user-item-interaction.parquet",
    columns=["user_id", "parent_asin", "review_rating", "recency_weight"]
)

# %%
# Build item text corpus: title + description + features
def build_text(row):
    parts = [str(row["item_title"] or "")]
    desc = row["description"]
    feats = row["features"]
    if isinstance(desc, list):
        parts += [str(d) for d in desc if d]
    if isinstance(feats, list):
        parts += [str(f) for f in feats if f]
    return " ".join(parts)

meta_df["text"] = meta_df.apply(build_text, axis=1)
meta_df = meta_df[["parent_asin", "item_title", "main_category", "text"]].reset_index(drop=True)

# Bring in is_free from item feature store
items_df = pd.read_parquet("../data/output/item.parquet", columns=["parent_asin", "is_free"])
meta_df = meta_df.merge(items_df, on="parent_asin", how="left")
meta_df["is_free"] = meta_df["is_free"].fillna(False)
meta_df

# %%
# TF-IDF item matrix — one row per item
tfidf = TfidfVectorizer(max_features=10_000, ngram_range=(1, 2), sublinear_tf=True)
item_matrix = tfidf.fit_transform(meta_df["text"])

# Stable index mapping
item_to_idx = {asin: i for i, asin in enumerate(meta_df["parent_asin"])}
idx_to_item = {i: asin for asin, i in item_to_idx.items()}

print(f"Item matrix: {item_matrix.shape}")

# %%
# Build user profile: weighted average of TF-IDF vectors for rated items
# weight = normalized rating (so higher-rated items influence profile more)
def build_user_profile(user_id):
    hist = user_item_itera[user_item_itera["user_id"] == user_id].copy()
    hist = hist[(hist["parent_asin"].isin(item_to_idx)) & (hist["review_rating"] >= 3)]
    if hist.empty:
        return None

    weights = (hist["review_rating"] / 5).values * hist["recency_weight"].values
    indices = hist["parent_asin"].map(item_to_idx).values

    weighted_sum = weights @ item_matrix[indices]                 # (1 x features)
    profile = weighted_sum / (weights.sum() + 1e-9)
    return np.asarray(profile).reshape(1, -1)


# Recommend top-N items for a user (excludes already-seen items)
def recommend_for_user(user_id, n=10):
    profile = build_user_profile(user_id)
    if profile is None:
        print(f"No history found for {user_id}")
        return pd.DataFrame()

    scores = cosine_similarity(profile, item_matrix).flatten()

    seen = set(user_item_itera.loc[user_item_itera["user_id"] == user_id, "parent_asin"])
    seen_idx = [item_to_idx[a] for a in seen if a in item_to_idx]
    scores[seen_idx] = -1

    # Compute user preferences from positive history
    hist = user_item_itera[
        (user_item_itera["user_id"] == user_id) &
        (user_item_itera["review_rating"] >= 3) &
        (user_item_itera["parent_asin"].isin(item_to_idx))
    ]
    hist_meta = hist.merge(meta_df[["parent_asin", "main_category", "is_free"]], on="parent_asin", how="left")
    top_category = hist_meta["main_category"].value_counts().index[0] if not hist_meta.empty else None
    prefers_free  = hist_meta["is_free"].mean() > 0.5 if not hist_meta.empty else False

    # Get a larger pool, re-rank, then trim to n
    pool_idx = np.argsort(scores)[::-1][: n * 5]
    pool = pd.DataFrame({
        "parent_asin": [idx_to_item[i] for i in pool_idx],
        "score":       scores[pool_idx],
    }).merge(meta_df[["parent_asin", "item_title", "main_category", "is_free"]], on="parent_asin")

    # Boost items in the user's preferred category
    if top_category:
        pool["score"] += 0.1 * (pool["main_category"] == top_category).astype(float)

    # Filter out paid apps if the user mostly reviews free ones
    if prefers_free:
        pool = pool[pool["is_free"]]

    return pool.sort_values("score", ascending=False).head(n).reset_index(drop=True)


# Find items similar to a given item
def similar_items(parent_asin, n=10):
    if parent_asin not in item_to_idx:
        print(f"Unknown item: {parent_asin}")
        return pd.DataFrame()

    idx = item_to_idx[parent_asin]
    scores = cosine_similarity(item_matrix[idx], item_matrix).flatten()
    scores[idx] = -1

    top_idx = np.argsort(scores)[::-1][:n]

    return pd.DataFrame({
        "parent_asin": [idx_to_item[i] for i in top_idx],
        "score":       scores[top_idx],
    }).merge(meta_df[["parent_asin", "item_title", "main_category"]], on="parent_asin")

# %%
# Demo — user recommendations
some_user = user_item_itera[user_item_itera["review_rating"] >= 3]["user_id"].iloc[0]
print(f"Recommendations for {some_user}")
recommend_for_user(some_user, n=5)

# %%
# Demo — item similarity
some_item = meta_df["parent_asin"].iloc[0]
print(f"Items similar to {some_item}: {meta_df.loc[0, 'item_title']}")
similar_items(some_item, n=5)
