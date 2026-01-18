# Data Usage Guide - Which Tables for Which Algorithm

**Project:** Amazon Software Recommendation Engine
**Purpose:** Maps ETL outputs to their usage in Collaborative and Content-Based Filtering
**Date:** 2026-01-18

---

## Overview

After running the improved ETL pipeline, we have multiple output files. This guide explains which files are used for which recommendation approach.

---

## Collaborative Filtering (CF)

**Core idea:** "Users who liked similar items will like similar items"

### Primary Data Used

| File | What It Contains | How CF Uses It |
|------|------------------|----------------|
| `train_matrix.npz` | User x Item matrix with ratings (1-5) | Core input for SVD, ALS, matrix factorization |
| `train_implicit.npz` | User x Item matrix with 0/1 (interacted or not) | For implicit feedback models (BPR, WARP) |
| `train_weighted.npz` | User x Item matrix with time-decayed ratings | For time-aware CF (recent matters more) |
| `user_to_idx.pkl` | Maps user_id to matrix row number | Look up users in the matrix |
| `item_to_idx.pkl` | Maps parent_asin to matrix column number | Look up items in the matrix |

### Supporting Data

| File | How CF Uses It |
|------|----------------|
| `user_features.parquet` | `user_segment` column determines CF weight (power_user=0.7, one_time=0.2) |
| `item_features.parquet` | `popularity_segment` determines if CF is reliable for this item |

### CF Algorithm Flow

```
1. Load train_matrix (users x items)
2. Apply matrix factorization (SVD/ALS)
   -> Learn user embeddings (what each user "likes")
   -> Learn item embeddings (what each item "is like")
3. For a target user:
   -> Find similar users OR
   -> Multiply user embedding x item embeddings
   -> Rank items by predicted score
```

---

## Content-Based Filtering (CBF)

**Core idea:** "Recommend items similar to what the user already liked"

### Primary Data Used

| File | What It Contains | How CBF Uses It |
|------|------------------|-----------------|
| `item_features.parquet` | 26 features per product | Calculate item-item similarity |
| `user_features.parquet` | 22 features per user | Build user preference profile |
| `user_category_features.parquet` | User's category preferences | Match users to item categories |
| `item_category_features.parquet` | Item's category memberships | Category-based similarity |
| `train_interactions.parquet` | Full review data with text | TF-IDF on review text, descriptions |

### Key Features for CBF

**Item Features (for item similarity):**
```
- main_category, primary_category  -> Category matching
- price, is_free, price_bucket     -> Price-based filtering
- store                            -> Brand/developer similarity
- avg_rating, quality_score        -> Quality filtering
- num_features, num_descriptions   -> Content richness
```

**User Features (for preference matching):**
```
- avg_price_purchased, free_app_ratio  -> Price preferences
- positive_rating_ratio                -> Optimist vs pessimist
- user_segment                         -> Adjust recommendation strategy
```

**Category Features:**
```
- user_category_features: "User X loves Antivirus software"
- item_category_features: "Product Y is in Antivirus category"
-> Match them!
```

### CBF Algorithm Flow

```
1. Build item profiles:
   - TF-IDF vectors from descriptions/reviews
   - Category one-hot encoding
   - Price bucket encoding

2. Build user preference profile:
   - Average of item profiles they liked
   - Weight by rating and recency

3. For a target user:
   - Calculate similarity(user_profile, each_item_profile)
   - Rank by similarity score
```

---

## Visual Summary

```
+------------------------------------------------------------------+
|                    COLLABORATIVE FILTERING                        |
+------------------------------------------------------------------+
|                                                                   |
|   train_matrix.npz ------+                                        |
|   train_implicit.npz ----+---> Matrix Factorization (SVD/ALS)     |
|   train_weighted.npz ----+           |                            |
|                                      v                            |
|   user_to_idx.pkl -----------> User Embeddings                    |
|   item_to_idx.pkl -----------> Item Embeddings                    |
|                                      |                            |
|   user_features.parquet ----> user_segment (weighting)            |
|   item_features.parquet ----> popularity_segment (fallback)       |
|                                      |                            |
|                                      v                            |
|                              CF Predictions                       |
+------------------------------------------------------------------+

+------------------------------------------------------------------+
|                    CONTENT-BASED FILTERING                        |
+------------------------------------------------------------------+
|                                                                   |
|   item_features.parquet ----+                                     |
|      - category             |                                     |
|      - price                +---> Item Profile Vectors            |
|      - store                |           |                         |
|      - quality_score        |           |                         |
|                             |           |                         |
|   item_category_features ---+           |                         |
|                                         v                         |
|   train_interactions.parquet --> TF-IDF on text --> Text Vectors  |
|                                         |                         |
|                                         v                         |
|   user_features.parquet ----+    Item Similarity Matrix           |
|      - price preferences    |           |                         |
|      - rating patterns      |           |                         |
|                             |           |                         |
|   user_category_features ---+---> User Preference Profile         |
|                                         |                         |
|                                         v                         |
|                              CBF Predictions                      |
+------------------------------------------------------------------+
```

---

## Hybrid: How They Combine

```python
# Pseudocode for hybrid recommendation

def recommend(user_id, n=10):
    # Get user segment
    user_seg = user_features[user_id]['user_segment']

    # Set weights based on user activity
    if user_seg == 'power_user':      # >10 reviews
        cf_weight, cbf_weight = 0.7, 0.3
    elif user_seg == 'occasional':    # 2-10 reviews
        cf_weight, cbf_weight = 0.5, 0.5
    else:                             # 1 review (cold start)
        cf_weight, cbf_weight = 0.2, 0.8

    # Get predictions from both
    cf_scores = collaborative_filter(user_id)    # Uses sparse matrices
    cbf_scores = content_based_filter(user_id)   # Uses feature tables

    # For cold-start ITEMS, override to pure CBF
    for item in items:
        if item_features[item]['popularity_segment'] == 'cold_start':
            cf_scores[item] = 0  # No CF data for this item

    # Combine
    final_scores = cf_weight * cf_scores + cbf_weight * cbf_scores

    return top_n(final_scores, n)
```

---

## Quick Reference Table

| File | CF | CBF | Purpose |
|------|:--:|:---:|---------|
| `train_matrix.npz` | Yes | - | Explicit ratings for matrix factorization |
| `train_implicit.npz` | Yes | - | Binary interactions for implicit models |
| `train_weighted.npz` | Yes | - | Time-decayed ratings |
| `user_to_idx.pkl` | Yes | - | Map user_id to matrix index |
| `item_to_idx.pkl` | Yes | - | Map item_id to matrix index |
| `user_features.parquet` | Yes | Yes | User segments, preferences |
| `item_features.parquet` | Yes | Yes | Item popularity, categories, price |
| `user_category_features.parquet` | - | Yes | User category preferences |
| `item_category_features.parquet` | - | Yes | Item category memberships |
| `train_interactions.parquet` | - | Yes | Full text for TF-IDF |
| `val_interactions.parquet` | Yes | Yes | Hyperparameter tuning |
| `test_interactions.parquet` | Yes | Yes | Final evaluation |

---

## Data Flow Summary

### For a New User (Cold Start)

```
User has 0-1 reviews
       |
       v
CBF weight = 0.8, CF weight = 0.2
       |
       v
Use: user_category_features (if any)
     item_features (categories, price)
     item_category_features
       |
       v
Recommend similar items to what they viewed/purchased
```

### For an Active User

```
User has 10+ reviews
       |
       v
CF weight = 0.7, CBF weight = 0.3
       |
       v
Use: train_matrix (find similar users)
     user embeddings from SVD/ALS
       |
       v
Recommend what similar users liked
```

### For a New Item (Cold Start)

```
Item has 0-1 reviews
       |
       v
Cannot use CF (no interaction data)
       |
       v
Use: item_features (category, price, store)
     item_category_features
     TF-IDF on description
       |
       v
Find similar items, recommend to users who liked those
```

---

## File Locations

After running the ETL pipeline, files are saved to:

```
data/processed/
|
+-- Enriched Data
|   +-- review_data_enriched.parquet / .csv
|   +-- merged_data_enriched.parquet / .csv
|
+-- Feature Tables
|   +-- user_features.parquet / .csv
|   +-- item_features.parquet / .csv
|   +-- user_category_features.parquet / .csv
|   +-- item_category_features.parquet / .csv
|
+-- Train/Val/Test Splits
|   +-- train_interactions.parquet / .csv
|   +-- val_interactions.parquet / .csv
|   +-- test_interactions.parquet / .csv
|
+-- ID Mappings
|   +-- user_to_idx.pkl / .csv
|   +-- item_to_idx.pkl / .csv
|
+-- Sparse Matrices
|   +-- train_matrix.npz
|   +-- train_implicit.npz
|   +-- train_weighted.npz
|
+-- Metadata
    +-- processing_summary.json
```

---

## Loading Examples

### Load for Collaborative Filtering

```python
import pickle
from scipy.sparse import load_npz

# Load sparse matrix
train_matrix = load_npz('data/processed/train_matrix.npz')

# Load ID mappings
with open('data/processed/user_to_idx.pkl', 'rb') as f:
    user_to_idx = pickle.load(f)
with open('data/processed/item_to_idx.pkl', 'rb') as f:
    item_to_idx = pickle.load(f)

# Reverse mappings (index -> ID)
idx_to_user = {v: k for k, v in user_to_idx.items()}
idx_to_item = {v: k for k, v in item_to_idx.items()}

print(f"Matrix shape: {train_matrix.shape}")
print(f"Sparsity: {100 * (1 - train_matrix.nnz / (train_matrix.shape[0] * train_matrix.shape[1])):.2f}%")
```

### Load for Content-Based Filtering

```python
import pandas as pd

# Load feature tables
item_features = pd.read_parquet('data/processed/item_features.parquet')
user_features = pd.read_parquet('data/processed/user_features.parquet')
user_cat = pd.read_parquet('data/processed/user_category_features.parquet')
item_cat = pd.read_parquet('data/processed/item_category_features.parquet')

# Load interactions for text features
train_df = pd.read_parquet('data/processed/train_interactions.parquet')

print(f"Items: {len(item_features)}, Users: {len(user_features)}")
print(f"Item features: {list(item_features.columns)}")
```

---

## References

- **ETL Pipeline:** See `ETL_IMPROVEMENT_GUIDE.md`
- **Data Schema:** See `data/DATA_DICTIONARY.md`
- **EDA Insights:** See `data/AMAZON_SOFTWARE_ANALYSIS_REPORT.md`
- **Project Context:** See `CLAUDE.md`

---

**Version:** 1.0
**Last Updated:** 2026-01-18
**Status:** Ready for model development
