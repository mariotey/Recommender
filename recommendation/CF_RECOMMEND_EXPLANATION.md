# Collaborative Filtering — How It Works

Collaborative filtering answers the question: **"What have users similar to this one liked?"**
It uses patterns in user-item interactions — not item content — to surface recommendations.

---

## Step 1: Load the Data

Two tables are loaded:
- **`meta_data.parquet`** — product catalog (used only to resolve ASINs to readable titles at the end)
- **`user-item-interaction.parquet`** — the full review history with signals like rating, recency, helpful votes, and price

---

## Step 2: Build an Interaction Score

Raw ratings alone are a weak signal. Instead, we combine **eight features** per review into a single composite score:

| Feature | What it captures |
|---|---|
| `review_rating` | Explicit user satisfaction |
| `recency_weight` | How recent the review is (newer = more relevant) |
| `helpful_vote` | Whether the review was found useful (quality signal) |
| `review_word_count` | Depth of engagement — longer reviews suggest more investment |
| `num_review_img` | User attached images → stronger engagement |
| `num_item_img` | Item quality signal (more images = better-presented product) |
| `num_item_videos` | Additional item quality signal |
| `price` | Inverted — lower price is treated as a positive signal |

**Process:**
1. All features are **Min-Max scaled** to [0, 1] so no single feature dominates
2. `price` is **inverted** (`1 − scaled_price`) so free/cheap items score higher
3. The **unweighted mean** of all eight features becomes the `interaction` score
4. Multiple reviews for the same (user, item) pair are averaged into one score

Result: a single float per (user, item) pair summarising the strength of that interaction.

---

## Step 3: Build a Sparse User-Item Matrix

With millions of (user, item, score) triples, we convert to a **Compressed Sparse Row (CSR) matrix**:

```
Shape: (2,589,466 users × 89,246 items)
Stored elements: 4,828,480
Data type: float32
```

Most entries are zero — only the 4.8 M observed interactions are stored. CSR format makes matrix algebra efficient even at this scale.

Integer indices are used internally; two lookup dictionaries (`idx_to_user`, `idx_to_item`) map back to original IDs.

---

## Step 4: Train the ALS Model

We use **Alternating Least Squares (ALS)** via the `implicit` library — a matrix factorization method designed for implicit feedback data.

**How ALS works:**
- It decomposes the user-item matrix into two low-rank matrices:
  - **User factors** — a vector of `k` latent dimensions per user
  - **Item factors** — a vector of `k` latent dimensions per item
- The dot product of a user vector and an item vector predicts their affinity
- ALS alternates between fixing item factors and optimising user factors, then vice versa — each step is a closed-form least-squares solve

**Hyperparameters used:**

| Parameter | Value | Effect |
|---|---|---|
| `factors` | 50 | Latent dimension size — captures 50 hidden taste dimensions |
| `regularization` | 0.01 | L2 penalty to prevent overfitting on sparse data |
| `iterations` | 20 | Number of alternating passes |

---

## Step 5: Generate Recommendations

To recommend for a user:
1. Look up the user's integer index
2. Pass the user's row from the sparse matrix (their interaction history) to `model.recommend()`
3. ALS scores every uninteracted item by computing the dot product of the user's latent vector with each item's latent vector
4. Top-N items (by score) are returned, already excluding items the user has seen

The returned ASINs are joined back to `meta_data` to surface human-readable titles and metadata.

> Example output for user `ae2222frpdmnomyomcwiantxp7uq`:
> Kindle for Mac, Calculator Plus Free, Flow Free, Max (HBO), AccuWeather

---

## Summary

```
user-item-interaction.parquet
      │
      ▼
  Select 8 features per review
  Min-Max normalise all features
  Invert price (lower price = better)
  Mean → interaction score
  Aggregate per (user, item)
      │
      ▼
  CSR Sparse Matrix  (2.6M users × 89k items)
      │
      ▼
  ALS Training  (factors=50, reg=0.01, iters=20)
  ├── User factor matrix  (2.6M × 50)
  └── Item factor matrix  (89k × 50)
      │
      ▼
  For a given user:
  dot(user_vector, item_matrix) → affinity scores
  Exclude seen items → Top-N recommendations
      │
      ▼
  Join ASINs → meta_data → readable titles
```

---

## Why Collaborative Filtering (vs Content-Based)?

| | Collaborative Filtering | Content-Based |
|--|--|--|
| Uses | Other users' behaviour | Item attributes (text, category) |
| Discovers surprising items? | Yes — finds cross-category gems | No — stays within known tastes |
| Works for new items? | No (cold start — needs interactions) | Yes |
| Works for new users? | No (needs history) | Yes (one liked item is enough) |
| Strength | Leverages crowd wisdom | Works in sparse/cold environments |

In this project CF is most reliable for **power users** (>10 reviews, ~1.2% of users) who generate 12.1% of all reviews. For the long tail of one-time users, content-based filtering takes over.

---

## Why ALS over Basic Matrix Factorization?

- **Implicit feedback:** Our interaction scores are derived signals (not explicit 1-5 star inputs), making ALS a natural fit — it treats the matrix as confidence-weighted preferences, not direct ratings
- **Scalability:** ALS parallelises well across a 2.6 M × 89 k matrix
- **Stability:** Regularisation prevents latent vectors from collapsing on sparse rows
