Collaborative Filtering (CF)
                                                                                                                                                                                                            
  Core idea: "Users who liked similar items will like similar items"

  Primary Data Used
  ┌────────────────────┬─────────────────────────────────────────────────┬───────────────────────────────────────────────┐
  │        File        │                What It Contains                 │                How CF Uses It                 │
  ├────────────────────┼─────────────────────────────────────────────────┼───────────────────────────────────────────────┤
  │ train_matrix.npz   │ User × Item matrix with ratings (1-5)           │ Core input for SVD, ALS, matrix factorization │
  ├────────────────────┼─────────────────────────────────────────────────┼───────────────────────────────────────────────┤
  │ train_implicit.npz │ User × Item matrix with 0/1 (interacted or not) │ For implicit feedback models (BPR, WARP)      │
  ├────────────────────┼─────────────────────────────────────────────────┼───────────────────────────────────────────────┤
  │ train_weighted.npz │ User × Item matrix with time-decayed ratings    │ For time-aware CF (recent matters more)       │
  ├────────────────────┼─────────────────────────────────────────────────┼───────────────────────────────────────────────┤
  │ user_to_idx.pkl    │ Maps user_id → matrix row number                │ Look up users in the matrix                   │
  ├────────────────────┼─────────────────────────────────────────────────┼───────────────────────────────────────────────┤
  │ item_to_idx.pkl    │ Maps parent_asin → matrix column number         │ Look up items in the matrix                   │
  └────────────────────┴─────────────────────────────────────────────────┴───────────────────────────────────────────────┘
  Supporting Data
  ┌───────────────────────┬─────────────────────────────────────────────────────────────────────────┐
  │         File          │                             How CF Uses It                              │
  ├───────────────────────┼─────────────────────────────────────────────────────────────────────────┤
  │ user_features.parquet │ user_segment column determines CF weight (power_user=0.7, one_time=0.2) │
  ├───────────────────────┼─────────────────────────────────────────────────────────────────────────┤
  │ item_features.parquet │ popularity_segment determines if CF is reliable for this item           │
  └───────────────────────┴─────────────────────────────────────────────────────────────────────────┘
  CF Algorithm Flow

  1. Load train_matrix (users × items)
  2. Apply matrix factorization (SVD/ALS)
     → Learn user embeddings (what each user "likes")
     → Learn item embeddings (what each item "is like")
  3. For a target user:
     → Find similar users OR
     → Multiply user embedding × item embeddings
     → Rank items by predicted score

  ---
  Content-Based Filtering (CBF)

  Core idea: "Recommend items similar to what the user already liked"

  Primary Data Used
  ┌────────────────────────────────┬─────────────────────────────┬─────────────────────────────────────┐
  │              File              │      What It Contains       │           How CBF Uses It           │
  ├────────────────────────────────┼─────────────────────────────┼─────────────────────────────────────┤
  │ item_features.parquet          │ 26 features per product     │ Calculate item-item similarity      │
  ├────────────────────────────────┼─────────────────────────────┼─────────────────────────────────────┤
  │ user_features.parquet          │ 22 features per user        │ Build user preference profile       │
  ├────────────────────────────────┼─────────────────────────────┼─────────────────────────────────────┤
  │ user_category_features.parquet │ User's category preferences │ Match users to item categories      │
  ├────────────────────────────────┼─────────────────────────────┼─────────────────────────────────────┤
  │ item_category_features.parquet │ Item's category memberships │ Category-based similarity           │
  ├────────────────────────────────┼─────────────────────────────┼─────────────────────────────────────┤
  │ train_interactions.parquet     │ Full review data with text  │ TF-IDF on review text, descriptions │
  └────────────────────────────────┴─────────────────────────────┴─────────────────────────────────────┘
  Key Features for CBF

  Item Features (for item similarity):
  - main_category, primary_category  → Category matching
  - price, is_free, price_bucket     → Price-based filtering
  - store                            → Brand/developer similarity
  - avg_rating, quality_score        → Quality filtering
  - num_features, num_descriptions   → Content richness

  User Features (for preference matching):
  - avg_price_purchased, free_app_ratio  → Price preferences
  - positive_rating_ratio                → Optimist vs pessimist
  - user_segment                         → Adjust recommendation strategy

  Category Features:
  - user_category_features: "User X loves Antivirus software"
  - item_category_features: "Product Y is in Antivirus category"
  → Match them!

  CBF Algorithm Flow

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

  ---
  Visual Summary

  ┌─────────────────────────────────────────────────────────────────┐
  │                    COLLABORATIVE FILTERING                       │
  ├─────────────────────────────────────────────────────────────────┤
  │                                                                  │
  │   train_matrix.npz ──────┐                                       │
  │   train_implicit.npz ────┼──→ Matrix Factorization (SVD/ALS)     │
  │   train_weighted.npz ────┘           │                           │
  │                                      ▼                           │
  │   user_to_idx.pkl ──────────→ User Embeddings                    │
  │   item_to_idx.pkl ──────────→ Item Embeddings                    │
  │                                      │                           │
  │   user_features.parquet ────→ user_segment (weighting)           │
  │   item_features.parquet ────→ popularity_segment (fallback)      │
  │                                      │                           │
  │                                      ▼                           │
  │                              CF Predictions                      │
  └─────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────┐
  │                    CONTENT-BASED FILTERING                       │
  ├─────────────────────────────────────────────────────────────────┤
  │                                                                  │
  │   item_features.parquet ────┐                                    │
  │      - category             │                                    │
  │      - price                ├──→ Item Profile Vectors            │
  │      - store                │           │                        │
  │      - quality_score        │           │                        │
  │                             │           │                        │
  │   item_category_features ───┘           │                        │
  │                                         ▼                        │
  │   train_interactions.parquet ──→ TF-IDF on text ──→ Text Vectors │
  │                                         │                        │
  │                                         ▼                        │
  │   user_features.parquet ────┐    Item Similarity Matrix          │
  │      - price preferences    │           │                        │
  │      - rating patterns      │           │                        │
  │                             │           │                        │
  │   user_category_features ───┴──→ User Preference Profile         │
  │                                         │                        │
  │                                         ▼                        │
  │                              CBF Predictions                     │
  └─────────────────────────────────────────────────────────────────┘

  ---
  Hybrid: How They Combine

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

  ---
  Quick Reference Table
  ┌────────────────────────────────┬─────┬─────┬───────────────────────────────────────────┐
  │              File              │ CF  │ CBF │                  Purpose                  │
  ├────────────────────────────────┼─────┼─────┼───────────────────────────────────────────┤
  │ train_matrix.npz               │ ✅  │ ❌  │ Explicit ratings for matrix factorization │
  ├────────────────────────────────┼─────┼─────┼───────────────────────────────────────────┤
  │ train_implicit.npz             │ ✅  │ ❌  │ Binary interactions for implicit models   │
  ├────────────────────────────────┼─────┼─────┼───────────────────────────────────────────┤
  │ train_weighted.npz             │ ✅  │ ❌  │ Time-decayed ratings                      │
  ├────────────────────────────────┼─────┼─────┼───────────────────────────────────────────┤
  │ user_to_idx.pkl                │ ✅  │ ❌  │ Map user_id to matrix index               │
  ├────────────────────────────────┼─────┼─────┼───────────────────────────────────────────┤
  │ item_to_idx.pkl                │ ✅  │ ❌  │ Map item_id to matrix index               │
  ├────────────────────────────────┼─────┼─────┼───────────────────────────────────────────┤
  │ user_features.parquet          │ ✅  │ ✅  │ User segments, preferences                │
  ├────────────────────────────────┼─────┼─────┼───────────────────────────────────────────┤
  │ item_features.parquet          │ ✅  │ ✅  │ Item popularity, categories, price        │
  ├────────────────────────────────┼─────┼─────┼───────────────────────────────────────────┤
  │ user_category_features.parquet │ ❌  │ ✅  │ User category preferences                 │
  ├────────────────────────────────┼─────┼─────┼───────────────────────────────────────────┤
  │ item_category_features.parquet │ ❌  │ ✅  │ Item category memberships                 │
  ├────────────────────────────────┼─────┼─────┼───────────────────────────────────────────┤
  │ train_interactions.parquet     │ ❌  │ ✅  │ Full text for TF-IDF                      │
  ├────────────────────────────────┼─────┼─────┼───────────────────────────────────────────┤
  │ val_interactions.parquet       │ ✅  │ ✅  │ Hyperparameter tuning                     │
  ├────────────────────────────────┼─────┼─────┼───────────────────────────────────────────┤
  │ test_interactions.parquet      │ ✅  │ ✅  │ Final evaluation                          │
  └────────────────────────────────┴─────┴─────┴───────────────────────────────────────────┘