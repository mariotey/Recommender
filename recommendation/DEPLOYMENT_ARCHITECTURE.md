# Deployment Architecture — Local + Supabase

This document describes how the recommender system is split between local/cloud compute
and a Supabase (Postgres) database for both the content-based and collaborative filtering pipelines.

---

## Core Principle: Transactional vs Analytical

The key design decision driving this architecture:

| Phase | Operation type | Best store |
|---|---|---|
| Training / ETL | Analytical — scan all rows, aggregate | Parquet files (local) |
| Serving / Inference | Transactional — lookup by `user_id` | Supabase Postgres (indexed) |
| Model artifacts | Read-once at startup | Local files (later: blob storage) |

The `user-item-interaction.parquet` (4.8M rows) is too large to load per request.
Instead it lives in Supabase, and at serving time we query **only the rows for the requesting user**.

---

## What Lives Where

```
Local machine / compute
├── src/content_based_filtering/     ← inference code
├── src/collaborative_filtering/     ← inference code
└── models/
    ├── cb_tfidf.joblib              ← TF-IDF vectorizer        (CB)
    ├── cb_item_matrix.npz           ← sparse item matrix       (CB)
    ├── cb_meta.parquet              ← item metadata            (CB)
    ├── als_model.npz                ← user + item factors      (CF)
    ├── cf_user_to_idx.json          ← user_id → matrix index   (CF)
    └── cf_idx_to_item.json          ← matrix index → asin      (CF)

Supabase (Postgres)
└── user_item_interactions table
    ├── user_id          TEXT   (indexed)
    ├── parent_asin      TEXT
    ├── review_rating    FLOAT
    └── recency_weight   FLOAT
```

Training artifacts (parquet files in `data/output/`) stay local — they are only needed
to rebuild models, not to serve recommendations.

---

## Content-Based Filtering — Serving Flow

### What the model needs at inference

| Artifact | Size (approx) | Loaded |
|---|---|---|
| `cb_tfidf.joblib` | ~5 MB | Once at startup |
| `cb_item_matrix.npz` | ~50–100 MB sparse | Once at startup |
| `cb_meta.parquet` | ~10 MB | Once at startup |
| User's interaction rows | Avg ~1.9 rows | Per request (Supabase) |

### Request lifecycle

```
GET /recommend/user?user_id=X
        │
        ▼
1. [Startup — cached]
   Load cb_tfidf, item_matrix, meta_df from disk into memory

        │
        ▼
2. [Per request — Supabase query]
   SELECT user_id, parent_asin, review_rating, recency_weight
   FROM user_item_interactions
   WHERE user_id = 'X'
   → returns ~1-5 rows in <5ms (indexed)

        │
        ▼
3. [In memory]
   build_user_profile()
   → weighted mean of item TF-IDF vectors (rating × recency_weight)
   → produces a (1 × 10,000) profile vector

        │
        ▼
4. [In memory]
   cosine_similarity(profile, item_matrix)
   → scores all 89k items

        │
        ▼
5. [In memory]
   Exclude seen items, apply category boost, free-item filter
   → return top-N as list[dict]
```

### Why Supabase works here

- 67.3% of users have exactly 1 review → Postgres returns 1 row instantly
- `user_id` index makes the lookup O(log n) regardless of table size
- Supabase free tier handles 4.8M rows comfortably (500MB limit, this table is ~200MB)

---

## Collaborative Filtering — Serving Flow

### What the model needs at inference

Refer to [CF_RECOMMEND_EXPLANATION.md](CF_RECOMMEND_EXPLANATION.md) for full training details.

| Artifact | Size (approx) | Loaded |
|---|---|---|
| User factor matrix (2.6M × 50) | ~500 MB | Once at startup |
| Item factor matrix (89k × 50) | ~17 MB | Once at startup |
| `cf_user_to_idx.json` | ~80 MB | Once at startup |
| `cf_idx_to_item.json` | ~5 MB | Once at startup |
| User's seen items | Avg ~1.9 rows | Per request (Supabase) |

> **Note:** The user factor matrix at 500MB is large for a serverless function.
> For local serving this is fine. For cloud deployment, this will need a persistent
> server (not a cold-start serverless function) or a vector store.

### Request lifecycle

```
GET /recommend/user?user_id=X
        │
        ▼
1. [Startup — cached]
   Load ALS user factors, item factors, lookup dicts from disk

        │
        ▼
2. [In memory — instant]
   user_idx = user_to_idx.get(user_id)
   if user_idx is None → cold start user → fallback to CB

        │
        ▼
3. [Per request — Supabase query]
   SELECT parent_asin FROM user_item_interactions
   WHERE user_id = 'X'
   → fetch seen ASINs to exclude from results

        │
        ▼
4. [In memory]
   user_vector = user_factors[user_idx]          # shape (50,)
   scores = item_factors @ user_vector            # dot product → (89k,)
   scores[seen_indices] = -inf                    # mask seen items

        │
        ▼
5. [In memory]
   top_idx = argsort(scores)[::-1][:n]
   asins = [idx_to_item[i] for i in top_idx]
   join to meta_df → return top-N as list[dict]
```

### Why Supabase is still needed for CF

The ALS model internally tracks user factors by integer index — it does not store
the list of items a user has already seen. To exclude seen items from recommendations,
we still need to query the user's interaction history from Supabase.

---

## Hybrid Routing (Future)

When both models are live, a router selects the blend based on user history depth:

```python
user_review_count = len(user_rows)   # from the Supabase query above

if user_review_count > 10:
    cf_weight, cb_weight = 0.7, 0.3   # power user
elif user_review_count >= 2:
    cf_weight, cb_weight = 0.5, 0.5   # occasional user
else:
    cf_weight, cb_weight = 0.2, 0.8   # one-time / cold start
```

Both models query the **same Supabase table** in the same request — no duplicate calls,
the result is reused for both profile building (CB) and seen-item exclusion (CF).

---

## Supabase Table Schema

```sql
CREATE TABLE user_item_interactions (
    user_id        TEXT    NOT NULL,
    parent_asin    TEXT    NOT NULL,
    review_rating  FLOAT   NOT NULL,
    recency_weight FLOAT   NOT NULL
);

-- Required for fast per-user lookups
CREATE INDEX idx_user_item_user_id ON user_item_interactions (user_id);
```

### Loading data (one-time ETL)

```python
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

df = pd.read_parquet("data/output/user-item-interaction.parquet",
                     columns=["user_id", "parent_asin", "review_rating", "recency_weight"])

conn = psycopg2.connect("postgresql://...")   # Supabase connection string
cur = conn.cursor()

rows = list(df.itertuples(index=False, name=None))
execute_values(cur,
    "INSERT INTO user_item_interactions (user_id, parent_asin, review_rating, recency_weight) VALUES %s",
    rows, page_size=10_000)

conn.commit()
```

---

## Summary

```
Training time (local, one-off)
  parquet files → fit models → save artifacts to models/
  parquet files → ETL → load into Supabase

Serving time (per request)
  Startup:     load model artifacts from disk into memory (once)
  Per request: query Supabase for user's rows (~1-5ms)
               run inference in memory (~10-50ms)
               return top-N recommendations
```

The parquet files are **training inputs only** — they never touch the serving path.
Supabase handles the transactional lookup; local memory handles the heavy math.
