# Feature Store Description
**Amazon Software Recommender System — Processed Data Tables**
**Last Updated:** 2026-03-28

---

## Overview

Three processed parquet tables form the feature store for the recommendation system:

| Table | File | Rows | Columns | Primary Key |
|-------|------|------|---------|-------------|
| Users | `data/output/user.parquet` | 2,589,466 | 25 | `user_id` |
| Items | `data/output/item.parquet` | 89,246 | 28 | `parent_asin` |
| Interactions | `data/output/user-item-interaction.parquet` | 4,880,181 | 40+ | `user_id` + `parent_asin` |

### Table Relationships
```
user.parquet ─── user_id ───┐
                             ├──► user-item-interaction.parquet
item.parquet ─── parent_asin┘
```
The interaction table is the fact table; users and items are dimension tables with pre-computed features.

---

## 1. `user.parquet` — User Feature Table

**Shape:** 2,589,466 users × 25 features

### Schema

| Column | Type | Description | Example | Stats |
|--------|------|-------------|---------|-------|
| `user_id` | string | Unique user identifier (join key) | `ae22236afrrsmqikgg7tptb75qea` | 2,589,466 unique |
| `num_reviews` | int64 | Total reviews written | 9 | mean=1.88, max=371 |
| `avg_rating_given` | float64 | Mean rating given across all reviews | 3.44 | mean=3.88, std=1.41 |
| `rating_std` | float64 | Std dev of ratings given (NaN for 1-review users) | 1.88 | mean=0.79, null for 1.74M users |
| `min_rating_given` | float64 | Lowest rating ever given | 1.0 | mean=3.62 |
| `max_rating_given` | float64 | Highest rating ever given | 5.0 | mean=4.08 |
| `first_review_date` | datetime | Date of earliest review | 2012-11-22 | range: 1999–2023 |
| `last_review_date` | datetime | Date of most recent review | 2018-12-15 | mean last: 2016-12 |
| `total_recency_weight` | float64 | Weighted sum of review recency (higher = more recent) | 0.009 | mean=0.043, max=49.1 |
| `total_helpful_votes_received` | int64 | Total helpful votes across all reviews | 79 | mean=9.28, max=11,514 |
| `avg_helpful_votes_per_review` | float64 | Avg helpful votes per review | 8.78 | mean=4.39, max=6,728 |
| `num_verified_purchases` | int64 | Count of verified purchase reviews | 9 | mean=1.79 |
| `avg_review_length` | float64 | Average character length of reviews | 260.6 | mean=144.2, max=33,125 |
| `avg_review_words` | float64 | Average word count of reviews | 50.7 | mean=27.1, max=6,225 |
| `num_review_img_sum` | int64 | Total images attached to reviews | 0 | mean=0.004, max=67 |
| `extreme_rating_ratio` | float64 | Fraction of reviews rated 1 or 5 stars | 0.78 | mean=0.70 |
| `positive_rating_ratio` | float64 | Fraction of reviews rated 4–5 stars | 0.67 | mean=0.71 |
| `negative_rating_ratio` | float64 | Fraction of reviews rated 1–2 stars | 0.33 | mean=0.21 |
| `avg_price_purchased` | float64 | Mean price of purchased items (-1 = free/unknown) | 0.55 | mean=2.45, max=1,699 |
| `free_app_ratio` | float64 | Fraction of reviews on free products | 0.33 | mean=0.72 |
| `days_active` | int64 | Days between first and last review | 2215 | mean=200, max=8,122 |
| `reviews_per_day` | float64 | Review rate (reviews / days_active) | 0.004 | mean=0.82 |
| `verified_purchase_ratio` | float64 | Fraction of reviews marked verified | 1.0 | mean=0.94 |
| `user_segment` | category | User type: `one_time`, `occasional`, `power` | occasional | one_time=1,743,452; occasional=~814K; power=~32K |
| `is_discriminating` | bool | True if user gives varied ratings (not all same score) | True | False=2,311,576 (89.3%) |

### Key Distributions
- **67.3%** are `one_time` users → cold start challenge
- **~31.4%** are `occasional` (2–10 reviews)
- **~1.2%** are `power` (>10 reviews)
- `rating_std` is NaN for all one-time users (846,014 have valid std)

---

## 2. `item.parquet` — Item Feature Table

**Shape:** 89,246 items × 28 features

### Schema

| Column | Type | Description | Example | Notes |
|--------|------|-------------|---------|-------|
| `parent_asin` | string | Product identifier (join key) | `0005162092` | 89,246 unique |
| `num_reviews` | int64 | Total review count | 2 | |
| `avg_rating` | float64 | Mean rating received | 5.0 | |
| `rating_std` | float64 | Std dev of ratings received | 0.0 | NaN for 1-review items |
| `min_rating` | float64 | Lowest rating received | 5.0 | |
| `max_rating` | float64 | Highest rating received | 5.0 | |
| `first_review_date` | datetime | Date of first review | 2013-06-08 | |
| `last_review_date` | datetime | Date of most recent review | 2015-02-02 | |
| `total_recency_weight` | float64 | Weighted recency score | 0.000218 | |
| `total_helpful_votes` | int64 | Total helpful votes on product reviews | 0 | |
| `avg_helpful_votes` | float64 | Avg helpful votes per review | 0.0 | |
| `num_verified_reviews` | int64 | Count of verified purchase reviews | 2 | |
| `positive_review_ratio` | float64 | Fraction rated 4–5 stars | 1.0 | |
| `negative_review_ratio` | float64 | Fraction rated 1–2 stars | 0.0 | |
| `main_category` | string | Top-level category | `software` | mostly "software" |
| `categories` | string | JSON list of subcategories | `[]` | many are empty |
| `store` | string | Seller/publisher name | `parsons technology, inc.` | |
| `price` | float64 | Item price in USD (-1 = unknown/free) | -1.0 | 18% have real price |
| `num_item_img` | int64 | Number of product images | 6 | |
| `num_item_videos` | int64 | Number of product videos | 0 | |
| `num_categories` | int64 | Depth of category hierarchy | 0 | |
| `is_free` | bool | True if item is free | False | ~61% free |
| `price_bucket` | category | Price range bucket | `10-25`, `free`, NaN | NaN when no price |
| `days_on_platform` | int64 | Days from first to last review | 605 | |
| `reviews_per_day` | float64 | Review velocity | 0.003 | |
| `verified_review_ratio` | float64 | Fraction of reviews that are verified | 1.0 | |
| `popularity_segment` | category | `cold_start`, `low_coverage`, `medium`, `popular` | `low_coverage` | |
| `quality_score` | float64 | Composite quality score (0–1) | 0.342 | 0.0 for cold_start items |

### Key Distributions (from CLAUDE.md)
- **cold_start** (≤1 review): 24,641 items (27.6%)
- **low_coverage** (2–10 reviews): 43,085 items (48.3%)
- **medium** (11–100 reviews): 16,603 items (18.6%)
- **popular** (>100 reviews): 4,922 items (5.5%)

---

## 3. `user-item-interaction.parquet` — Interaction / Fact Table

**Shape:** 4,880,181 interactions × 40+ columns
**Note:** Too large to load fully into memory at once — use chunked reading or column selection.

### Schema

#### Interaction Columns (core)
| Column | Type | Description |
|--------|------|-------------|
| `review_id` | int64 | Unique review identifier |
| `user_id` | string | User identifier (FK → user.parquet) |
| `parent_asin` | string | Product identifier (FK → item.parquet) |
| `asin` | string | Specific product variant ASIN |
| `review_datetime` | datetime | Timestamp of review |
| `review_year` | int32 | Year of review |
| `review_month` | int32 | Month of review |
| `days_since_review` | int64 | Days from review to dataset cutoff (for recency) |
| `recency_weight` | float64 | Time-decay weight for this review |

#### Review Signal Columns
| Column | Type | Description |
|--------|------|-------------|
| `review_rating` | float64 | Star rating 1–5 (main feedback signal) |
| `verified_purchase` | bool | Whether this was a verified purchase |
| `helpful_vote` | int64 | Number of helpful votes received |
| `is_extreme_rating` | bool | True if rating is 1 or 5 |
| `is_positive` | bool | True if rating ≥ 4 |
| `is_negative` | bool | True if rating ≤ 2 |
| `review_text` | string | Full review text |
| `review_title` | string | Review title |
| `review_text_length` | int64 | Character length of review text |
| `review_title_length` | int64 | Character length of review title |
| `review_word_count` | int64 | Word count of review text |
| `num_review_img` | int64 | Images attached to this review |
| `review_images` | string | Review image URLs (JSON) |

#### Item Context Columns (denormalized)
| Column | Type | Description |
|--------|------|-------------|
| `item_title` | string | Product title |
| `main_category` | string | Product category |
| `categories` | list[string] | Subcategory list |
| `description` | list[string] | Product description paragraphs |
| `features` | list[string] | Product feature bullets |
| `details` | string | Product details (JSON) |
| `item_images` | struct | Product images (hi_res, large, thumb, variant) |
| `item_videos` | struct | Product videos (title, url, user_id) |
| `item_rating` | float64 | Overall item rating at time of dataset |
| `store` | string | Seller/publisher |
| `price` | float64 | Price (-1 = unknown) |
| `num_item_img` | int64 | Number of product images |
| `num_item_videos` | int64 | Number of product videos |
| `is_free` | bool | Whether item is free |
| `has_price_info` | bool | Whether price data exists |
| `price_bucket` | category | Price range bucket |
| `num_categories` | int64 | Category depth |

---

## Usage Examples

### Load Dimension Tables (safe — fits in memory)
```python
import pandas as pd

users = pd.read_parquet('data/output/user.parquet')
items = pd.read_parquet('data/output/item.parquet')
```

### Load Interaction Table — Select Columns Only
```python
# Only load what you need (full table may OOM)
interactions = pd.read_parquet(
    'data/output/user-item-interaction.parquet',
    columns=['user_id', 'parent_asin', 'review_rating', 'review_datetime',
             'verified_purchase', 'helpful_vote', 'recency_weight', 'is_positive']
)
```

### Build User-Item Matrix
```python
from scipy.sparse import csr_matrix

interactions = pd.read_parquet(
    'data/output/user-item-interaction.parquet',
    columns=['user_id', 'parent_asin', 'review_rating']
)

user_ids = users['user_id'].reset_index(drop=True)
item_ids = items['parent_asin'].reset_index(drop=True)

user_to_idx = {u: i for i, u in enumerate(user_ids)}
item_to_idx = {p: i for i, p in enumerate(item_ids)}

row = interactions['user_id'].map(user_to_idx)
col = interactions['parent_asin'].map(item_to_idx)
data = interactions['review_rating'].values

matrix = csr_matrix((data, (row, col)), shape=(len(user_ids), len(item_ids)))
# Shape: (2,589,466 × 89,246), density ~0.002%
```

### Temporal Train/Test Split
```python
interactions = pd.read_parquet(
    'data/output/user-item-interaction.parquet',
    columns=['user_id', 'parent_asin', 'review_rating', 'review_datetime', 'recency_weight']
)

train = interactions[interactions['review_datetime'] < '2022-01-01']
val   = interactions[(interactions['review_datetime'] >= '2022-01-01') &
                     (interactions['review_datetime'] < '2023-01-01')]
test  = interactions[interactions['review_datetime'] >= '2023-01-01']
```

### Join User/Item Features to Interactions
```python
df = interactions.merge(users, on='user_id', how='left')
df = df.merge(items, on='parent_asin', how='left', suffixes=('_interaction', '_item'))
```

### Filter by User Segment
```python
power_users = users[users['user_segment'] == 'power']['user_id']
interactions_power = interactions[interactions['user_id'].isin(power_users)]
```

---

## Notes for Recommendation Modeling

1. **Primary rating signal:** `review_rating` in interactions (1–5 stars)
2. **Implicit feedback alternative:** `is_positive` / `verified_purchase` as binary signals
3. **Recency weighting:** Use `recency_weight` column — already computed with time decay
4. **Cold start routing:**
   - `user_segment == 'one_time'` → content-based or popularity
   - `popularity_segment == 'cold_start'` → content-based (no CF)
5. **Text features:** `review_text`, `description`, `features` available in interaction table for TF-IDF/embeddings
6. **Quality weighting:** `helpful_vote` and `verified_purchase` can up-weight trustworthy reviews
7. **Price features:** `is_free` (61% of items) is more reliable than `price` (only 18% have valid price)
