# ETL Improvement Guide - User & Item Features for Hybrid Recommender

**Project:** Amazon Software Recommendation Engine
**Purpose:** Comprehensive guide for improving user and item feature extraction
**Date:** 2026-01-12
**Based On:** Analysis of `etl_notebook/get_user_item_df.ipynb`

---

---

## Critical Issues in Current ETL

### 🔴 Issue 1: Dropping NaN Prices Loses Data

**Current Code:**
```python
merged_df = (
    review_df
    .merge(meta_df, how="left", on="parent_asin")
    .dropna(subset=["price"])  # ❌ PROBLEMATIC
)
```

**Problem:**
- Removes **420,279 reviews** (8.6% of dataset)
- Eliminates products without price info (20.47% of products)
- Loses valuable interaction data for collaborative filtering

**Solution:**
```python
# Keep all data, handle NaN appropriately
merged_df['price'] = merged_df['price'].fillna(-1)  # -1 = unknown
merged_df['is_free'] = merged_df['price'] == 0.0
merged_df['has_price_info'] = merged_df['price'] >= 0
```

---

### 🔴 Issue 2: Category Explosion Before Aggregation

**Current Code:**
```python
merged_df = merged_df.explode("categories").rename(columns={"categories": "category"})

# Then aggregate
user_df = merged_df.groupby(["user_id", "main_category", "category"])...
```

**Problem:**
- Creates **duplicate rows** for items with multiple categories
- A review counted once becomes counted 3× if item has 3 categories
- **Inflates all count-based metrics** (num_reviews, etc.)
- User/item dataframes have multiple rows per user/item (not ideal for features)

**Solution:**
```python
# DON'T explode before aggregation
# Keep categories as array in main dataframe
# Create separate user-category and item-category tables AFTER aggregation
```

---

### 🔴 Issue 3: Missing Temporal Features

**Current Code:**
- Uses `timestamp` but doesn't convert to datetime
- Only tracks `max` review date, not `min`
- No recency weighting (critical for 24-year dataset!)

**Solution:**
```python
# Convert timestamp to datetime
review_df['review_date'] = pd.to_datetime(review_df['timestamp'], unit='ms')

# Calculate recency weight (exponential decay)
max_date = review_df['review_date'].max()
review_df['days_since_review'] = (max_date - review_df['review_date']).dt.days
review_df['recency_weight'] = np.exp(-review_df['days_since_review'] / 365.25)
```

---

### 🔴 Issue 4: Limited Feature Engineering

**Current Code:**
- Only basic aggregations (count, mean, max)
- No text features (review length, word count)
- No quality signals (helpful vote aggregates)
- No user/item segmentation

**Solution:** See comprehensive feature engineering below.

---

## Improved ETL Pipeline

### Step 1: Data Loading & Preprocessing

```python
import numpy as np
import pandas as pd
import json
from datetime import datetime
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

# -------------------- DATA LOADING --------------------
review_df = pd.read_parquet("../data/output/review_data.parquet")
meta_df = pd.read_parquet("../data/output/meta_data.parquet")

# Add review_id as column (not index)
review_df = review_df.reset_index(names="review_id")

# -------------------- PREPROCESSING --------------------
# Convert timestamp to datetime (critical for temporal features)
review_df['review_date'] = pd.to_datetime(review_df['timestamp'], unit='ms')

# Extract text features
review_df['review_text_length'] = review_df['text'].str.len()
review_df['review_title_length'] = review_df['title'].str.len()
review_df['review_word_count'] = review_df['text'].str.split().str.len()

# Sentiment proxy (simple heuristic)
review_df['is_extreme_rating'] = review_df['rating'].isin([1.0, 5.0])
review_df['is_positive'] = review_df['rating'] >= 4.0
review_df['is_negative'] = review_df['rating'] <= 2.0

# Image counts
review_df['num_review_images'] = review_df['images'].apply(
    lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0
)
```

---

### Step 2: Merge with Metadata

```python
# -------------------- MERGE WITH METADATA --------------------
# DON'T drop NaN prices!
merged_df = review_df.merge(meta_df, on='parent_asin', how='left')

# Handle price properly
merged_df['price'] = merged_df['price'].fillna(-1)  # -1 indicates unknown
merged_df['is_free'] = merged_df['price'] == 0.0
merged_df['has_price_info'] = merged_df['price'] >= 0
merged_df['price_bucket'] = pd.cut(
    merged_df['price'],
    bins=[-1, 0, 10, 25, 50, 100, 1000, 2000],
    labels=['unknown', 'free', '0-10', '10-25', '25-50', '50-100', '100+']
)

# Extract metadata features
def safe_len(x):
    """Safely get length of arrays"""
    if isinstance(x, (list, np.ndarray)):
        return len(x)
    return 0

merged_df['num_features'] = merged_df['features'].apply(safe_len)
merged_df['num_descriptions'] = merged_df['description'].apply(safe_len)
merged_df['num_categories'] = merged_df['categories'].apply(safe_len)

# Extract primary category (first in list)
merged_df['primary_category'] = merged_df['categories'].apply(
    lambda x: x[0] if isinstance(x, list) and len(x) > 0 else 'unknown'
)
```

---

### Step 3: Temporal Features

```python
# -------------------- TEMPORAL FEATURES --------------------
# Calculate days since review (from most recent date in dataset)
max_date = merged_df['review_date'].max()
merged_df['days_since_review'] = (max_date - merged_df['review_date']).dt.days
merged_df['review_year'] = merged_df['review_date'].dt.year
merged_df['review_month'] = merged_df['review_date'].dt.month

# Recency weight (exponential decay with 1-year half-life)
merged_df['recency_weight'] = np.exp(-merged_df['days_since_review'] / 365.25)
```

---

### Step 4: Train/Validation/Test Split

```python
# -------------------- TRAIN/VAL/TEST SPLIT --------------------
# Temporal split (as recommended in CLAUDE.md)
train_df = merged_df[merged_df['review_date'] < '2022-01-01']
val_df = merged_df[
    (merged_df['review_date'] >= '2022-01-01') &
    (merged_df['review_date'] < '2023-01-01')
]
test_df = merged_df[merged_df['review_date'] >= '2023-01-01']

print(f"Train: {len(train_df)} reviews ({train_df['review_date'].min()} to {train_df['review_date'].max()})")
print(f"Val: {len(val_df)} reviews")
print(f"Test: {len(test_df)} reviews")
```

---

## User Features

### Goal: ONE ROW PER USER

Create a comprehensive user profile that captures behavior, preferences, and quality signals.

```python
# -------------------- USER FEATURES --------------------
# Build comprehensive user profile (ONE ROW PER USER)

user_features = (
    train_df[train_df['verified_purchase'] == True]  # Only verified for training
    .groupby('user_id')
    .agg({
        # Basic stats
        'review_id': 'count',
        'rating': ['mean', 'std', 'min', 'max'],

        # Temporal
        'review_date': ['min', 'max'],
        'recency_weight': 'sum',

        # Quality signals
        'helpful_vote': ['sum', 'mean'],
        'verified_purchase': 'sum',

        # Text engagement
        'review_text_length': 'mean',
        'review_word_count': 'mean',
        'num_review_images': 'sum',

        # Rating patterns
        'is_extreme_rating': 'mean',
        'is_positive': 'mean',
        'is_negative': 'mean',

        # Price sensitivity
        'price': 'mean',
        'is_free': 'mean'
    })
)

# Flatten multi-level columns
user_features.columns = ['_'.join(col).strip('_') for col in user_features.columns]
user_features = user_features.reset_index()

# Rename for clarity
user_features = user_features.rename(columns={
    'review_id_count': 'num_reviews',
    'rating_mean': 'avg_rating_given',
    'rating_std': 'rating_std',
    'rating_min': 'min_rating_given',
    'rating_max': 'max_rating_given',
    'review_date_min': 'first_review_date',
    'review_date_max': 'last_review_date',
    'recency_weight_sum': 'total_recency_weight',
    'helpful_vote_sum': 'total_helpful_votes_received',
    'helpful_vote_mean': 'avg_helpful_votes_per_review',
    'verified_purchase_sum': 'num_verified_purchases',
    'review_text_length_mean': 'avg_review_length',
    'review_word_count_mean': 'avg_review_words',
    'num_review_images_sum': 'total_review_images',
    'is_extreme_rating_mean': 'extreme_rating_ratio',
    'is_positive_mean': 'positive_rating_ratio',
    'is_negative_mean': 'negative_rating_ratio',
    'price_mean': 'avg_price_purchased',
    'is_free_mean': 'free_app_ratio'
})

# Derive additional features
user_features['days_active'] = (
    user_features['last_review_date'] - user_features['first_review_date']
).dt.days + 1

user_features['reviews_per_day'] = (
    user_features['num_reviews'] / user_features['days_active']
)

user_features['verified_purchase_ratio'] = (
    user_features['num_verified_purchases'] / user_features['num_reviews']
)

# User segmentation (from EDA)
user_features['user_segment'] = pd.cut(
    user_features['num_reviews'],
    bins=[0, 1, 10, np.inf],
    labels=['one_time', 'occasional', 'power_user']
)

# Rating discriminativeness (higher std = more discriminating)
user_features['is_discriminating'] = user_features['rating_std'] > 1.0

print(f"Total users: {len(user_features)}")
print(f"User segments:\n{user_features['user_segment'].value_counts()}")
```

### User Features Summary

| Feature | Description | Use Case |
|---------|-------------|----------|
| `num_reviews` | Total reviews by user | User activity level |
| `avg_rating_given` | Average rating given | User leniency bias |
| `rating_std` | Std dev of ratings | Discriminativeness |
| `first_review_date` | First review date | User tenure |
| `last_review_date` | Most recent review | Recency of activity |
| `total_recency_weight` | Sum of recency weights | Time-weighted engagement |
| `total_helpful_votes_received` | Total helpful votes | Review quality |
| `avg_helpful_votes_per_review` | Avg helpful votes | Review influence |
| `num_verified_purchases` | Count of verified reviews | Authenticity signal |
| `avg_review_length` | Avg review text length | Engagement level |
| `avg_review_words` | Avg word count | Verbosity |
| `total_review_images` | Total images uploaded | Visual engagement |
| `extreme_rating_ratio` | % of 1 or 5 star ratings | Rating extremity |
| `positive_rating_ratio` | % of 4-5 star ratings | Positivity bias |
| `negative_rating_ratio` | % of 1-2 star ratings | Negativity bias |
| `avg_price_purchased` | Avg price of reviewed items | Price preference |
| `free_app_ratio` | % of free apps reviewed | Free vs paid preference |
| `days_active` | Days between first/last review | Tenure |
| `reviews_per_day` | Review frequency | Activity rate |
| `verified_purchase_ratio` | % verified purchases | Trustworthiness |
| `user_segment` | one_time/occasional/power_user | User type for weighting |
| `is_discriminating` | Boolean: std > 1.0 | Rating variance flag |

---

## Item Features

### Goal: ONE ROW PER ITEM

Create a comprehensive product profile capturing popularity, quality, and engagement.

```python
# -------------------- ITEM FEATURES --------------------
# Build comprehensive item profile (ONE ROW PER ITEM)

item_features = (
    train_df
    .groupby('parent_asin')
    .agg({
        # Basic stats
        'user_id': 'count',
        'rating': ['mean', 'std', 'min', 'max'],

        # Temporal
        'review_date': ['min', 'max'],
        'recency_weight': 'sum',

        # Quality signals
        'helpful_vote': ['sum', 'mean', 'max'],
        'verified_purchase': 'sum',

        # Engagement
        'review_text_length': 'mean',
        'review_word_count': 'mean',

        # Rating patterns
        'is_positive': 'mean',
        'is_negative': 'mean',

        # Metadata (take first/mode)
        'main_category': 'first',
        'primary_category': 'first',
        'store': 'first',
        'price': 'first',
        'is_free': 'first',
        'num_features': 'first',
        'num_descriptions': 'first',
        'num_categories': 'first'
    })
)

# Flatten columns
item_features.columns = ['_'.join(col).strip('_') for col in item_features.columns]
item_features = item_features.reset_index()

# Rename
item_features = item_features.rename(columns={
    'user_id_count': 'num_reviews',
    'rating_mean': 'avg_rating',
    'rating_std': 'rating_std',
    'rating_min': 'min_rating',
    'rating_max': 'max_rating',
    'review_date_min': 'first_review_date',
    'review_date_max': 'last_review_date',
    'recency_weight_sum': 'total_recency_weight',
    'helpful_vote_sum': 'total_helpful_votes',
    'helpful_vote_mean': 'avg_helpful_votes',
    'helpful_vote_max': 'max_helpful_votes',
    'verified_purchase_sum': 'num_verified_reviews',
    'review_text_length_mean': 'avg_review_length_received',
    'review_word_count_mean': 'avg_review_words_received',
    'is_positive_mean': 'positive_review_ratio',
    'is_negative_mean': 'negative_review_ratio'
})

# Clean up column names (remove '_first')
for col in item_features.columns:
    if col.endswith('_first'):
        item_features = item_features.rename(columns={col: col.replace('_first', '')})

# Derive features
item_features['days_on_platform'] = (
    item_features['last_review_date'] - item_features['first_review_date']
).dt.days + 1

item_features['reviews_per_day'] = (
    item_features['num_reviews'] / item_features['days_on_platform']
)

item_features['verified_review_ratio'] = (
    item_features['num_verified_reviews'] / item_features['num_reviews']
)

# Item popularity segment (from EDA)
item_features['popularity_segment'] = pd.cut(
    item_features['num_reviews'],
    bins=[0, 1, 10, 100, np.inf],
    labels=['cold_start', 'low_coverage', 'medium', 'popular']
)

# Quality score (Wilson lower bound for rating confidence)
def wilson_lower_bound(pos, n, confidence=0.95):
    """Wilson score interval for rating confidence"""
    if n == 0:
        return 0
    z = 1.96  # 95% confidence
    phat = pos / n
    return (phat + z*z/(2*n) - z * np.sqrt((phat*(1-phat)+z*z/(4*n))/n))/(1+z*z/n)

item_features['quality_score'] = item_features.apply(
    lambda row: wilson_lower_bound(
        row['positive_review_ratio'] * row['num_reviews'],
        row['num_reviews']
    ),
    axis=1
)

print(f"Total items: {len(item_features)}")
print(f"Popularity segments:\n{item_features['popularity_segment'].value_counts()}")
```

### Item Features Summary

| Feature | Description | Use Case |
|---------|-------------|----------|
| `num_reviews` | Total reviews received | Popularity signal |
| `avg_rating` | Average rating | Quality signal |
| `rating_std` | Std dev of ratings | Rating consistency |
| `first_review_date` | First review received | Product age |
| `last_review_date` | Most recent review | Recency |
| `total_recency_weight` | Sum of recency weights | Recent popularity |
| `total_helpful_votes` | Total helpful votes on reviews | Engagement |
| `avg_helpful_votes` | Avg helpful votes per review | Review quality |
| `max_helpful_votes` | Max helpful votes (single review) | Peak engagement |
| `num_verified_reviews` | Count of verified reviews | Authenticity |
| `avg_review_length_received` | Avg review text length | User engagement |
| `avg_review_words_received` | Avg word count | Detailed feedback |
| `positive_review_ratio` | % of positive reviews | Satisfaction rate |
| `negative_review_ratio` | % of negative reviews | Dissatisfaction rate |
| `main_category` | Primary category | Content-based filtering |
| `primary_category` | First category in list | Fine-grained category |
| `store` | Developer/seller name | Brand signal |
| `price` | Product price | Price-based filtering |
| `is_free` | Boolean: free product | Monetization model |
| `num_features` | Count of feature bullets | Description richness |
| `num_descriptions` | Count of description paragraphs | Content availability |
| `num_categories` | Count of categories | Multi-category flag |
| `days_on_platform` | Days between first/last review | Product lifetime |
| `reviews_per_day` | Review velocity | Growth rate |
| `verified_review_ratio` | % verified reviews | Trust score |
| `popularity_segment` | cold_start/low/medium/popular | Popularity tier |
| `quality_score` | Wilson lower bound score | Confidence-adjusted quality |

---

## User-Item Interaction Matrix

### Sparse Matrices for Collaborative Filtering

```python
# -------------------- USER-ITEM MATRIX --------------------
# Create sparse matrices for collaborative filtering

def create_interaction_matrix(df, value_col='rating'):
    """Create sparse user-item interaction matrix"""
    # Create mappings
    user_ids = df['user_id'].unique()
    item_ids = df['parent_asin'].unique()

    user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
    item_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}

    # Map to indices
    user_indices = df['user_id'].map(user_to_idx)
    item_indices = df['parent_asin'].map(item_to_idx)
    values = df[value_col].values

    # Create sparse matrix
    matrix = csr_matrix(
        (values, (user_indices, item_indices)),
        shape=(len(user_ids), len(item_ids))
    )

    return matrix, user_to_idx, item_to_idx

# Create explicit rating matrix
train_matrix, user_to_idx, item_to_idx = create_interaction_matrix(train_df)
print(f"Train matrix shape: {train_matrix.shape}")
print(f"Sparsity: {100 * (1 - train_matrix.nnz / (train_matrix.shape[0] * train_matrix.shape[1])):.4f}%")

# Create binary implicit feedback matrix (1 if interaction occurred)
train_df['implicit_feedback'] = 1
train_implicit, _, _ = create_interaction_matrix(train_df, value_col='implicit_feedback')

# Create weighted matrix (rating * recency_weight for temporal decay)
train_df['weighted_rating'] = train_df['rating'] * train_df['recency_weight']
train_weighted, _, _ = create_interaction_matrix(train_df, value_col='weighted_rating')

print(f"\n✅ Created 3 interaction matrices:")
print(f"  - train_matrix: Explicit ratings (1.0-5.0)")
print(f"  - train_implicit: Binary feedback (0 or 1)")
print(f"  - train_weighted: Time-decayed ratings")
```

### Matrix Types

1. **Explicit Rating Matrix** (`train_matrix`)
   - Values: 1.0 to 5.0
   - Use: Traditional collaborative filtering (SVD, ALS)

2. **Implicit Feedback Matrix** (`train_implicit`)
   - Values: 0 or 1 (interaction occurred)
   - Use: Implicit feedback models (BPR, WARP)

3. **Weighted Rating Matrix** (`train_weighted`)
   - Values: rating × recency_weight
   - Use: Time-aware collaborative filtering
   - Recent reviews have more weight

---

## Category Features

### Separate Tables for Many-to-Many Relationships

```python
# -------------------- CATEGORY FEATURES --------------------
# Handle categories separately (many-to-many relationship)

# User-Category preferences
user_category_features = (
    train_df
    .explode('categories')  # NOW we explode (after main aggregations)
    .groupby(['user_id', 'categories'])
    .agg({
        'review_id': 'count',
        'rating': 'mean'
    })
    .reset_index()
    .rename(columns={
        'categories': 'category',
        'review_id': 'num_reviews_in_category',
        'rating': 'avg_rating_in_category'
    })
)

# Calculate user's favorite categories
user_top_categories = (
    user_category_features
    .sort_values(['user_id', 'num_reviews_in_category'], ascending=[True, False])
    .groupby('user_id')
    .head(3)  # Top 3 categories per user
)

# Item-Category mapping (for content-based filtering)
item_category_features = (
    meta_df[['parent_asin', 'categories', 'main_category']]
    .explode('categories')
    .rename(columns={'categories': 'category'})
)

print(f"User-category pairs: {len(user_category_features)}")
print(f"Item-category pairs: {len(item_category_features)}")
print(f"Unique categories: {user_category_features['category'].nunique()}")
```

---

## Save Processed Data

```python
# -------------------- SAVE EVERYTHING --------------------
import os
import pickle
from scipy.sparse import save_npz

output_dir = "../data/processed/"
os.makedirs(output_dir, exist_ok=True)

# Helper function to save both parquet and CSV
def save_dataframe(df, name, output_dir=output_dir):
    """Save dataframe in both parquet and CSV formats"""
    parquet_path = f"{output_dir}{name}.parquet"
    csv_path = f"{output_dir}{name}.csv"

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)

    print(f"  Saved {name}: {len(df):,} rows")
    return parquet_path, csv_path

# -------------------- 1. ENRICHED SOURCE DATA --------------------
print("Saving enriched source data...")

# Enriched review dataframe (with text features, sentiment proxies, etc.)
save_dataframe(review_df, "review_data_enriched")

# Enriched merged dataframe (reviews + metadata with all derived features)
save_dataframe(merged_df, "merged_data_enriched")

# -------------------- 2. FEATURE DATAFRAMES --------------------
print("\nSaving feature dataframes...")

save_dataframe(user_features, "user_features")
save_dataframe(item_features, "item_features")
save_dataframe(user_category_features, "user_category_features")
save_dataframe(item_category_features, "item_category_features")

# -------------------- 3. TRAIN/VAL/TEST SPLITS --------------------
print("\nSaving train/val/test splits...")

save_dataframe(train_df, "train_interactions")
save_dataframe(val_df, "val_interactions")
save_dataframe(test_df, "test_interactions")

# -------------------- 4. ID MAPPINGS --------------------
print("\nSaving ID mappings...")

# Save as pickle (for Python)
with open(f"{output_dir}user_to_idx.pkl", 'wb') as f:
    pickle.dump(user_to_idx, f)
with open(f"{output_dir}item_to_idx.pkl", 'wb') as f:
    pickle.dump(item_to_idx, f)

# Also save as CSV for portability
user_mapping_df = pd.DataFrame([
    {'user_id': uid, 'user_idx': idx}
    for uid, idx in user_to_idx.items()
])
item_mapping_df = pd.DataFrame([
    {'parent_asin': iid, 'item_idx': idx}
    for iid, idx in item_to_idx.items()
])
user_mapping_df.to_csv(f"{output_dir}user_to_idx.csv", index=False)
item_mapping_df.to_csv(f"{output_dir}item_to_idx.csv", index=False)

print(f"  Saved user_to_idx: {len(user_to_idx):,} users (.pkl + .csv)")
print(f"  Saved item_to_idx: {len(item_to_idx):,} items (.pkl + .csv)")

# -------------------- 5. SPARSE MATRICES --------------------
print("\nSaving sparse matrices...")

save_npz(f"{output_dir}train_matrix.npz", train_matrix)
save_npz(f"{output_dir}train_implicit.npz", train_implicit)
save_npz(f"{output_dir}train_weighted.npz", train_weighted)

print(f"  Saved train_matrix.npz: {train_matrix.shape}")
print(f"  Saved train_implicit.npz: {train_implicit.shape}")
print(f"  Saved train_weighted.npz: {train_weighted.shape}")

# -------------------- 6. METADATA / SUMMARY --------------------
print("\nSaving processing metadata...")

# Save a summary of the processing for reproducibility
processing_summary = {
    'processing_date': datetime.now().isoformat(),
    'source_files': {
        'review_data': '../data/output/review_data.parquet',
        'meta_data': '../data/output/meta_data.parquet'
    },
    'dataset_stats': {
        'total_reviews': len(merged_df),
        'total_users': len(user_features),
        'total_items': len(item_features),
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'train_date_range': f"< 2022-01-01",
        'val_date_range': f"2022-01-01 to 2022-12-31",
        'test_date_range': f">= 2023-01-01"
    },
    'feature_counts': {
        'user_features': len(user_features.columns),
        'item_features': len(item_features.columns)
    },
    'matrix_stats': {
        'shape': train_matrix.shape,
        'nnz': train_matrix.nnz,
        'sparsity_pct': 100 * (1 - train_matrix.nnz / (train_matrix.shape[0] * train_matrix.shape[1]))
    }
}

with open(f"{output_dir}processing_summary.json", 'w') as f:
    json.dump(processing_summary, f, indent=2, default=str)

print(f"  Saved processing_summary.json")

# -------------------- FINAL SUMMARY --------------------
print("\n" + "="*60)
print("✅ ALL PROCESSED DATA SAVED TO:", output_dir)
print("="*60)

print(f"""
📁 Files created:

ENRICHED DATA (parquet + csv):
  - review_data_enriched     ({len(review_df):,} reviews)
  - merged_data_enriched     ({len(merged_df):,} reviews with metadata)

FEATURE TABLES (parquet + csv):
  - user_features            ({len(user_features):,} users, {len(user_features.columns)} features)
  - item_features            ({len(item_features):,} items, {len(item_features.columns)} features)
  - user_category_features   ({len(user_category_features):,} user-category pairs)
  - item_category_features   ({len(item_category_features):,} item-category pairs)

TRAIN/VAL/TEST SPLITS (parquet + csv):
  - train_interactions       ({len(train_df):,} reviews, <2022)
  - val_interactions         ({len(val_df):,} reviews, 2022)
  - test_interactions        ({len(test_df):,} reviews, 2023+)

ID MAPPINGS (pkl + csv):
  - user_to_idx              ({len(user_to_idx):,} users)
  - item_to_idx              ({len(item_to_idx):,} items)

SPARSE MATRICES (npz):
  - train_matrix             (explicit ratings)
  - train_implicit           (binary feedback)
  - train_weighted           (time-decayed ratings)

METADATA:
  - processing_summary.json  (reproducibility info)
""")
```

---

## Summary of Improvements

| Aspect | Current Implementation | Improved Implementation |
|--------|----------------------|------------------------|
| **Price handling** | Drops 420K reviews with NaN | Keeps all, fills NaN with -1 |
| **Category handling** | Explodes before aggregation (inflates counts) | Separate many-to-many table after aggregation |
| **User features** | Multiple rows per user (due to category explosion) | **ONE row per user** with 22 features |
| **Item features** | Multiple rows per item (due to category explosion) | **ONE row per item** with 26 features |
| **Temporal features** | Only max date captured | Full temporal analysis: min, max, recency weights, decay |
| **Text features** | None | Review length, word count, image uploads |
| **Quality signals** | Limited to helpful_vote | Helpful votes (sum/mean/max), verified ratio, Wilson score |
| **User segmentation** | None | One-time / Occasional / Power user (from EDA) |
| **Item segmentation** | None | Cold-start / Low / Medium / Popular (from EDA) |
| **Train/test split** | None | **Temporal split** (train: <2022, val: 2022, test: 2023) |
| **CF matrices** | None | 3 sparse matrices (explicit, implicit, weighted) |
| **Rating patterns** | Only mean rating | Mean, std, min, max, positive/negative ratios |
| **Engagement metrics** | None | Review frequency, activity tenure, reviews per day |
| **Price features** | Just average price | Price buckets, free ratio, price preferences |
| **Feature count** | User: 6, Item: 10 | **User: 22, Item: 26** |

---

## Feature Engineering Alignment with Hybrid Strategy

### For Collaborative Filtering

✅ **User-Item Interaction Matrices**
- Explicit ratings matrix (for SVD, ALS)
- Implicit feedback matrix (for BPR, WARP)
- Weighted ratings matrix (time-decayed)

✅ **User Activity Segmentation**
- Power users (>10 reviews) → CF weight: 0.7
- Occasional users (2-10 reviews) → CF weight: 0.5
- One-time users (1 review) → CF weight: 0.2

✅ **Item Popularity Segmentation**
- Popular items (>100 reviews) → Use CF
- Medium items (11-100 reviews) → Hybrid
- Low coverage (2-10 reviews) → More content-based
- Cold start (0-1 reviews) → Pure content-based

### For Content-Based Filtering

✅ **Item Content Features**
- Category hierarchy (main_category, primary_category, all categories)
- Price buckets for price-aware recommendations
- Product metadata richness (num_features, num_descriptions)
- Store/developer for brand-based recommendations

✅ **User Preference Features**
- Category preferences (user_category_features table)
- Price sensitivity (avg_price_purchased, free_app_ratio)
- Rating patterns (positive/negative ratios)

✅ **Text Content** (for future TF-IDF/embeddings)
- Review text (preserved in train/val/test splits)
- Product descriptions and features (in metadata)

---

## Next Steps

### Immediate (Data Preparation Complete)
1. ✅ User features created (22 features)
2. ✅ Item features created (26 features)
3. ✅ Sparse matrices for CF (3 types)
4. ✅ Temporal train/val/test split
5. ✅ Category preference tables

### Phase 2: Baseline Models
- [ ] Popularity-based recommender
- [ ] Item-based collaborative filtering (cosine similarity on sparse matrix)
- [ ] Content-based filtering (TF-IDF on descriptions + category matching)
- [ ] Evaluation framework (Precision@K, Recall@K, nDCG@K)

### Phase 3: Advanced Models
- [ ] Matrix Factorization (SVD, ALS on train_matrix)
- [ ] Neural Collaborative Filtering (user/item embeddings)
- [ ] Deep content models (BERT embeddings on text)
- [ ] Hybrid ensemble (dynamic weighting by user segment)

---

## References

- **Data Schema:** See `data/DATA_DICTIONARY.md`
- **EDA Insights:** See `data/AMAZON_SOFTWARE_ANALYSIS_REPORT.md`
- **Project Context:** See `CLAUDE.md`
- **Current ETL:** See `etl_notebook/get_user_item_df.ipynb`

---

**Version:** 1.0
**Last Updated:** 2026-01-12
**Status:** Ready for implementation
