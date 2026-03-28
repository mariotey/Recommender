# Content-Based Filtering — How It Works

Content-based filtering answers the question: **"Based on what this user liked, what similar items should we recommend?"**
It uses item attributes (text, category) — not other users' behaviour.

---

## Step 1: Load the Data

Three tables are loaded:
- **`meta_data.parquet`** — product catalog (title, description, features, category)
- **`item.parquet`** — item feature store (includes `is_free` flag)
- **`user-item-interaction.parquet`** — user review history (rating, recency weight)

---

## Step 2: Build an Item Description String

For each product, we combine its **title + description + features** into a single text string.

> Example for a productivity app:
> `"Focus Timer Pomodoro technique stay focused work in short bursts boost productivity"`

This gives us one text blob per item that captures what the product is about.

---

## Step 3: Convert Text to Numbers (TF-IDF)

Computers can't work with raw text, so we convert each item's text into a vector of numbers using **TF-IDF**:

- **TF (Term Frequency):** how often a word appears in this item's text
- **IDF (Inverse Document Frequency):** how rare that word is across all items — rare words get higher weight because they're more distinctive

Result: a matrix of shape `(89,246 items × 10,000 words)` where each row is a numeric fingerprint of an item's content.

---

## Step 4: Build a User Profile

To understand what a user likes, we look at items they rated **3 stars or above** (positive signal only) and compute a **weighted average** of those items' TF-IDF vectors.

**Weight for each item =** `(rating / 5) × recency_weight`

- Higher-rated items influence the profile more
- More recent ratings influence the profile more

The result is a single vector that represents the user's taste — a "centre of gravity" in item space.

> Example: if a user liked three productivity apps, their profile vector will point towards words like "focus", "timer", "tasks".

---

## Step 5: Find Similar Items (Cosine Similarity)

We measure how similar the user's profile vector is to every item's vector using **cosine similarity** — it compares the direction of two vectors, not their size.

- Score of **1.0** = perfect match in content
- Score of **0.0** = nothing in common

Items the user has already interacted with are excluded, and a pool of `5×N` candidates is fetched for re-ranking in the next step.

---

## Step 6: Re-rank by User Preferences (Category + Price)

Text similarity alone doesn't know whether a user prefers free apps or a specific category. So after getting the candidate pool, we apply two preference checks derived from the user's positive history:

**Preferred category boost:**
The user's most-reviewed category gets a `+0.1` score bonus on matching items. If a user mostly reviewed `software` apps, software items rank higher.

**Free app filter:**
If more than 50% of the user's liked items are free, paid apps are filtered out entirely — no point recommending a $49 app to someone who only uses free tools.

The pool is then re-sorted and trimmed to the final top-N.

---

## Step 7: Item-to-Item Similarity (Bonus)

The same cosine similarity can be run between **one item and all others**, giving us "items similar to X" — useful for "you might also like" sections on a product page.

---

## Summary

```
meta_data.parquet + item.parquet (is_free)
      │
      ▼
  Build text  ──►  TF-IDF  ──►  Item Matrix (89k × 10k)
                                        │
user-item-interaction.parquet           │
      │                                 │
      ▼                                 │
  Filter liked items (rating ≥ 3)       │
  Weight by rating × recency            │
  Weighted average  ──►  User Profile   │
                               │        │
                               ▼        ▼
                          Cosine Similarity
                               │
                               ▼
                       Candidate Pool (5×N)
                               │
                               ▼
                 Re-rank by User Preferences
                 ├── +0.1 boost for top category
                 └── filter paid if prefers free
                               │
                               ▼
                        Top-N Recommendations
```

## Why Content-Based (vs Collaborative Filtering)?

| | Content-Based | Collaborative Filtering |
|--|--|--|
| Uses | Item attributes (text, category) | Other users' behaviour |
| Works for new items? | Yes | No (needs interaction history) |
| Works for new users? | Yes (just need one liked item) | No (cold start problem) |
| Weakness | Can't discover surprising items | Fails with sparse data |

In this dataset, **27.6% of items have ≤1 review** and **67.3% of users have only 1 review** — making content-based filtering an essential component of the recommendation system.
