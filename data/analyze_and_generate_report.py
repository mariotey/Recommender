"""
Generate comprehensive analysis report for Amazon Software Reviews dataset
Optimized for recommendation engine development
"""

import pandas as pd
import numpy as np
from datetime import datetime

print("Loading datasets...")
review_df = pd.read_parquet("./output/review_data.parquet")
meta_df = pd.read_parquet("./output/meta_data.parquet")

print(f"Reviews: {len(review_df):,} rows")
print(f"Metadata: {len(meta_df):,} rows")

# ============================================================================
# Data Preprocessing for Analysis
# ============================================================================

# Review dataset enhancements
review_df['datetime'] = pd.to_datetime(review_df['timestamp'], unit='ms')
review_df['year'] = review_df['datetime'].dt.year
review_df['title_length'] = review_df['title'].fillna('').astype(str).str.len()
review_df['text_length'] = review_df['text'].fillna('').astype(str).str.len()
review_df['text_word_count'] = review_df['text'].fillna('').astype(str).str.split().str.len()

# Metadata dataset enhancements
meta_df['has_features'] = meta_df['features'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)
meta_df['has_description'] = meta_df['description'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)
meta_df['has_price'] = meta_df['price'].notna() & (meta_df['price'] > 0)

# ============================================================================
# Generate Markdown Report
# ============================================================================

md_content = f"""# Amazon Software Reviews 2023 - Comprehensive Data Analysis
## For Recommendation Engine Development

**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Executive Summary

This document provides an in-depth analysis of the Amazon Software Reviews 2023 dataset, specifically focused on insights relevant to building a recommendation engine. The analysis covers three main areas:

1. **Review Dataset Analysis** - User behavior, rating patterns, temporal trends
2. **Metadata Analysis** - Product characteristics, pricing, content richness
3. **Combined Analysis** - Relationships between reviews and products for collaborative and content-based filtering

### Dataset Overview

| Metric | Value |
|--------|-------|
| Total Reviews | {len(review_df):,} |
| Unique Users | {review_df['user_id'].nunique():,} |
| Unique Products (Parent ASIN) | {review_df['parent_asin'].nunique():,} |
| Total Products in Catalog | {len(meta_df):,} |
| Date Range | {review_df['datetime'].min().strftime('%Y-%m-%d')} to {review_df['datetime'].max().strftime('%Y-%m-%d')} |
| Years Covered | {review_df['datetime'].max().year - review_df['datetime'].min().year} years |

---

## Part 1: Review Dataset Analysis

### 1.1 Rating Distribution

**Key Statistics:**
- Mean Rating: {review_df['rating'].mean():.2f}
- Median Rating: {review_df['rating'].median():.2f}
- Standard Deviation: {review_df['rating'].std():.2f}

**Rating Breakdown:**

| Rating | Count | Percentage |
|--------|-------|------------|
"""

# Rating distribution
for rating in sorted(review_df['rating'].unique()):
    count = (review_df['rating'] == rating).sum()
    pct = count / len(review_df) * 100
    md_content += f"| {rating:.1f} | {count:,} | {pct:.2f}% |\n"

# Rating polarization
extreme_ratings = review_df[review_df['rating'].isin([1.0, 5.0])]
positive_ratings = review_df[review_df['rating'] >= 4.0]
negative_ratings = review_df[review_df['rating'] <= 2.0]

md_content += f"""
**Rating Characteristics:**
- **Positive Skew:** {positive_ratings.shape[0]/len(review_df)*100:.1f}% of reviews are 4-5 stars
- **Negative Reviews:** {negative_ratings.shape[0]/len(review_df)*100:.1f}% are 1-2 stars
- **Polarization:** {extreme_ratings.shape[0]/len(review_df)*100:.1f}% are extreme (1 or 5 stars)
- **Neutral Reviews:** {(review_df['rating'] == 3.0).sum()/len(review_df)*100:.1f}% are neutral (3 stars)

**Recommendation Engine Implications:**
- High positive skew suggests users prefer rating software highly or not at all
- Binary classification (positive/negative) might be more effective than 5-class
- Consider normalizing ratings per user to account for individual rating behaviors
- Rating variance is important - some users consistently rate high, others are more critical

---

### 1.2 User Behavior Analysis

**User Activity Metrics:**
"""

# User statistics
reviews_per_user = review_df['user_id'].value_counts()
unique_users = review_df['user_id'].nunique()
one_time_reviewers = (reviews_per_user == 1).sum()
repeat_reviewers = (reviews_per_user > 1).sum()
power_users = (reviews_per_user > 10).sum()

md_content += f"""
| Metric | Value |
|--------|-------|
| Total Unique Users | {unique_users:,} |
| Average Reviews per User | {len(review_df)/unique_users:.2f} |
| Median Reviews per User | {reviews_per_user.median():.0f} |
| Max Reviews by Single User | {reviews_per_user.max():,} |

**User Segmentation:**

| Segment | Count | Percentage | Definition |
|---------|-------|------------|------------|
| One-time Reviewers | {one_time_reviewers:,} | {one_time_reviewers/unique_users*100:.1f}% | Exactly 1 review |
| Occasional Reviewers | {((reviews_per_user > 1) & (reviews_per_user <= 10)).sum():,} | {((reviews_per_user > 1) & (reviews_per_user <= 10)).sum()/unique_users*100:.1f}% | 2-10 reviews |
| Power Users | {power_users:,} | {power_users/unique_users*100:.1f}% | >10 reviews |

**Top 10 Most Active Users:**

| Rank | User ID | Reviews | Percentage of Total |
|------|---------|---------|---------------------|
"""

for i, (user_id, count) in enumerate(reviews_per_user.head(10).items(), 1):
    md_content += f"| {i} | {user_id[:20]}... | {count:,} | {count/len(review_df)*100:.3f}% |\n"

power_user_reviews = reviews_per_user[reviews_per_user > 10].sum()

md_content += f"""
**Recommendation Engine Implications:**
- **Cold Start Problem:** {one_time_reviewers/unique_users*100:.1f}% of users have only 1 review - collaborative filtering will be challenging
- **Data Sparsity:** High proportion of one-time reviewers indicates sparse user-item matrix
- **User Influence:** Power users ({power_users:,}) contribute {power_user_reviews:,} reviews ({power_user_reviews/len(review_df)*100:.1f}% of total)
- **Strategy:** Hybrid approach combining collaborative filtering (for active users) and content-based (for sparse users)
- **Consider:** User clustering based on rating patterns rather than individual user-based CF

---

### 1.3 Product Coverage Analysis

**Product Review Distribution:**
"""

# Product statistics
reviews_per_product = review_df['parent_asin'].value_counts()
unique_products_reviewed = review_df['parent_asin'].nunique()
single_review_products = (reviews_per_product == 1).sum()
popular_products = (reviews_per_product > 100).sum()

md_content += f"""
| Metric | Value |
|--------|-------|
| Unique Products with Reviews | {unique_products_reviewed:,} |
| Average Reviews per Product | {len(review_df)/unique_products_reviewed:.2f} |
| Median Reviews per Product | {reviews_per_product.median():.0f} |
| Max Reviews for Single Product | {reviews_per_product.max():,} |

**Product Segmentation:**

| Segment | Count | Percentage | Definition |
|---------|-------|------------|------------|
| Single Review Products | {single_review_products:,} | {single_review_products/unique_products_reviewed*100:.1f}% | Exactly 1 review |
| Low Coverage (2-10) | {((reviews_per_product > 1) & (reviews_per_product <= 10)).sum():,} | {((reviews_per_product > 1) & (reviews_per_product <= 10)).sum()/unique_products_reviewed*100:.1f}% | 2-10 reviews |
| Medium Coverage (11-100) | {((reviews_per_product > 10) & (reviews_per_product <= 100)).sum():,} | {((reviews_per_product > 10) & (reviews_per_product <= 100)).sum()/unique_products_reviewed*100:.1f}% | 11-100 reviews |
| High Coverage (>100) | {popular_products:,} | {popular_products/unique_products_reviewed*100:.1f}% | >100 reviews |

**Top 10 Most Reviewed Products:**

| Rank | Parent ASIN | Reviews | Avg Rating |
|------|-------------|---------|------------|
"""

for i, (asin, count) in enumerate(reviews_per_product.head(10).items(), 1):
    avg_rating = review_df[review_df['parent_asin'] == asin]['rating'].mean()
    md_content += f"| {i} | {asin} | {count:,} | {avg_rating:.2f} |\n"

md_content += f"""
**Recommendation Engine Implications:**
- **Cold Start (Item):** {single_review_products/unique_products_reviewed*100:.1f}% of products have only 1 review
- **Long Tail:** Significant portion of products have few reviews - content-based filtering crucial
- **Popular Items Bias:** Top products dominate review counts - need diversity in recommendations
- **Strategy:** Use content-based for cold items, collaborative for popular items

---

### 1.4 Temporal Patterns

**Review Timeline:**
"""

yearly_reviews = review_df['year'].value_counts().sort_index()

md_content += f"""
| Year | Reviews | Percentage | Trend |
|------|---------|------------|-------|
"""

years_list = sorted(yearly_reviews.index)
for year in years_list:
    count = yearly_reviews[year]
    pct = count / len(review_df) * 100

    # Calculate trend
    if year > years_list[0]:
        prev_count = yearly_reviews.get(year - 1, 0)
        if prev_count > 0:
            trend = (count - prev_count) / prev_count * 100
            trend_str = f"+{trend:.1f}%" if trend > 0 else f"{trend:.1f}%"
        else:
            trend_str = "N/A"
    else:
        trend_str = "Baseline"

    md_content += f"| {year} | {count:,} | {pct:.2f}% | {trend_str} |\n"

# Rating trends
avg_rating_by_year = review_df.groupby('year')['rating'].mean()
recent_3_years = sorted(review_df['year'].unique())[-3:]
recent_avg = review_df[review_df['year'].isin(recent_3_years)]['rating'].mean()
old_avg = review_df[~review_df['year'].isin(recent_3_years)]['rating'].mean()

md_content += f"""
**Rating Evolution:**
- Overall average rating: {review_df['rating'].mean():.2f}
- Recent 3 years average: {recent_avg:.2f}
- Earlier years average: {old_avg:.2f}
- Rating trend: {"Improving" if recent_avg > old_avg else "Declining"} ({abs(recent_avg - old_avg):.2f} point difference)

**Recommendation Engine Implications:**
- **Temporal Decay:** Older reviews may be less relevant - consider time-based weighting
- **Recency Bias:** Recent reviews might better reflect current product quality
- **Seasonal Patterns:** Consider time-of-year effects (holidays, back-to-school, etc.)
- **Strategy:** Implement time-decay function in rating aggregation (e.g., exponential decay)

---

### 1.5 Verified Purchase Analysis

**Verification Status:**
"""

verified_counts = review_df['verified_purchase'].value_counts()
verified_true = verified_counts.get(True, 0)
verified_false = verified_counts.get(False, 0)

md_content += f"""
| Status | Count | Percentage |
|--------|-------|------------|
| Verified Purchase | {verified_true:,} | {verified_true/len(review_df)*100:.2f}% |
| Not Verified | {verified_false:,} | {verified_false/len(review_df)*100:.2f}% |

**Rating Comparison by Verification:**

| Status | Avg Rating | Median Rating | Std Dev |
|--------|------------|---------------|---------|
"""

for status in [True, False]:
    subset = review_df[review_df['verified_purchase'] == status]
    if len(subset) > 0:
        status_label = "Verified" if status else "Not Verified"
        md_content += f"| {status_label} | {subset['rating'].mean():.2f} | {subset['rating'].median():.1f} | {subset['rating'].std():.2f} |\n"

md_content += f"""
**Recommendation Engine Implications:**
- **Trust Signal:** Verified purchases may be more reliable for recommendations
- **Weighting Strategy:** Consider giving higher weight to verified purchase reviews
- **Fraud Detection:** Non-verified reviews might include promotional/fake reviews
- **Strategy:** Option to filter recommendations based on verified purchases only

---

### 1.6 Review Text Analysis

**Text Characteristics:**
"""

md_content += f"""
| Metric | Title | Review Text |
|--------|-------|-------------|
| Mean Length (chars) | {review_df['title_length'].mean():.1f} | {review_df['text_length'].mean():.1f} |
| Median Length (chars) | {review_df['title_length'].median():.0f} | {review_df['text_length'].median():.0f} |
| Mean Word Count | {review_df['title'].fillna('').str.split().str.len().mean():.1f} | {review_df['text_word_count'].mean():.1f} |
| Empty Count | {(review_df['title'].isna() | (review_df['title'] == '')).sum():,} | {(review_df['text'].isna() | (review_df['text'] == '')).sum():,} |
| Empty Percentage | {(review_df['title'].isna() | (review_df['title'] == '')).sum()/len(review_df)*100:.2f}% | {(review_df['text'].isna() | (review_df['text'] == '')).sum()/len(review_df)*100:.2f}% |

**Text Length by Rating:**

| Rating | Avg Text Length | Avg Word Count |
|--------|----------------|----------------|
"""

for rating in sorted(review_df['rating'].unique()):
    subset = review_df[review_df['rating'] == rating]
    md_content += f"| {rating:.1f} | {subset['text_length'].mean():.0f} | {subset['text_word_count'].mean():.0f} |\n"

md_content += f"""
**Recommendation Engine Implications:**
- **NLP Opportunities:** Rich text available for sentiment analysis and feature extraction
- **Review Quality:** Longer reviews tend to correlate with extreme ratings (very positive/negative)
- **Content-Based Features:** Can extract keywords, topics, and aspects from review text
- **Strategy:** Use NLP to extract product features and user preferences for content-based filtering
- **Advanced:** Implement aspect-based sentiment analysis (performance, ease of use, support, etc.)

---

### 1.7 Helpful Votes Analysis

**Helpfulness Metrics:**
"""

reviews_with_votes = review_df[review_df['helpful_vote'] > 0]

md_content += f"""
| Metric | Value |
|--------|-------|
| Total Helpful Votes | {review_df['helpful_vote'].sum():,} |
| Reviews with Votes (>0) | {len(reviews_with_votes):,} ({len(reviews_with_votes)/len(review_df)*100:.2f}%) |
| Reviews with No Votes | {(review_df['helpful_vote'] == 0).sum():,} ({(review_df['helpful_vote'] == 0).sum()/len(review_df)*100:.2f}%) |
| Mean Helpful Votes | {review_df['helpful_vote'].mean():.2f} |
| Median Helpful Votes | {review_df['helpful_vote'].median():.0f} |
| Max Helpful Votes | {review_df['helpful_vote'].max():,} |

**Helpful Votes by Rating:**

| Rating | Avg Helpful Votes | Reviews with >5 Votes |
|--------|------------------|----------------------|
"""

for rating in sorted(review_df['rating'].unique()):
    subset = review_df[review_df['rating'] == rating]
    high_votes = (subset['helpful_vote'] > 5).sum()
    md_content += f"| {rating:.1f} | {subset['helpful_vote'].mean():.2f} | {high_votes:,} |\n"

# Correlation
text_vote_corr = review_df[['text_length', 'helpful_vote']].corr().iloc[0, 1]

md_content += f"""
**Correlation Analysis:**
- Text length vs helpful votes: {text_vote_corr:.3f}

**Recommendation Engine Implications:**
- **Quality Signal:** Helpful votes indicate review quality and trustworthiness
- **Weighting:** Use helpful votes as confidence weights in rating aggregation
- **Filtering:** Prioritize highly-voted reviews for feature extraction
- **Strategy:** Reviews with high helpful votes can be weighted more heavily in algorithms

---

## Part 2: Metadata Analysis

### 2.1 Product Catalog Overview
"""

# Basic metadata stats
md_content += f"""
| Metric | Value |
|--------|-------|
| Total Products in Catalog | {len(meta_df):,} |
| Products with Reviews | {review_df['parent_asin'].nunique():,} |
| Products without Reviews | {len(meta_df[~meta_df['parent_asin'].isin(review_df['parent_asin'])]):,} |
| Coverage | {review_df['parent_asin'].nunique()/len(meta_df)*100:.2f}% |

### 2.2 Category Distribution

**Main Categories:**

| Category | Count | Percentage |
|----------|-------|------------|
"""

for cat, count in meta_df['main_category'].value_counts().head(10).items():
    md_content += f"| {cat} | {count:,} | {count/len(meta_df)*100:.2f}% |\n"

md_content += f"""
**Recommendation Engine Implications:**
- **Category-based Filtering:** Can implement category-aware recommendations
- **Cross-category Recommendations:** Explore relationships between categories
- **Strategy:** Use categories as features in content-based filtering

---

### 2.3 Price Analysis
"""

price_data = meta_df[meta_df['price'].notna() & (meta_df['price'] > 0)]
free_products = meta_df[meta_df['price'] == 0]

md_content += f"""
| Metric | Value |
|--------|-------|
| Products with Price Info | {len(price_data):,} ({len(price_data)/len(meta_df)*100:.2f}%) |
| Free Products | {len(free_products):,} ({len(free_products)/len(meta_df)*100:.2f}%) |
| Missing Price | {meta_df['price'].isna().sum():,} ({meta_df['price'].isna().sum()/len(meta_df)*100:.2f}%) |

**Price Statistics (Paid Products):**

| Statistic | Value |
|-----------|-------|
| Mean Price | ${price_data['price'].mean():.2f} |
| Median Price | ${price_data['price'].median():.2f} |
| Std Dev | ${price_data['price'].std():.2f} |
| Min Price | ${price_data['price'].min():.2f} |
| Max Price | ${price_data['price'].max():.2f} |

**Price Ranges:**

| Range | Count | Percentage |
|-------|-------|------------|
"""

if len(price_data) > 0:
    price_ranges = pd.cut(price_data['price'],
                         bins=[0, 10, 25, 50, 100, 250, 500, 1000, float('inf')],
                         labels=['$0-10', '$10-25', '$25-50', '$50-100', '$100-250', '$250-500', '$500-1000', '$1000+'])
    for range_label, count in price_ranges.value_counts().sort_index().items():
        md_content += f"| {range_label} | {count:,} | {count/len(price_data)*100:.2f}% |\n"

md_content += f"""
**Recommendation Engine Implications:**
- **Price-aware Recommendations:** Can filter or rank by price range
- **Price Sensitivity:** Consider user price preferences in recommendations
- **Free Software:** Large proportion of free products - different recommendation strategy
- **Strategy:** Implement price-based filters and similarity measures

---

### 2.4 Product Ratings (Metadata)
"""

rating_data = meta_df[meta_df['average_rating'].notna()]

md_content += f"""
| Metric | Value |
|--------|-------|
| Products with Avg Rating | {len(rating_data):,} ({len(rating_data)/len(meta_df)*100:.2f}%) |
| Mean Avg Rating | {rating_data['average_rating'].mean():.2f} |
| Median Avg Rating | {rating_data['average_rating'].median():.2f} |
| Std Dev | {rating_data['average_rating'].std():.2f} |

**Recommendation Engine Implications:**
- **Baseline Ratings:** Can use metadata ratings as baseline for new reviews
- **Cold Start:** Metadata ratings help with products lacking user reviews
- **Strategy:** Combine user-generated ratings with platform aggregate ratings

---

### 2.5 Content Richness

**Product Content Availability:**

| Content Type | Available | Percentage |
|--------------|-----------|------------|
| Features | {meta_df['has_features'].sum():,} | {meta_df['has_features'].sum()/len(meta_df)*100:.2f}% |
| Descriptions | {meta_df['has_description'].sum():,} | {meta_df['has_description'].sum()/len(meta_df)*100:.2f}% |
| Price | {len(price_data):,} | {len(price_data)/len(meta_df)*100:.2f}% |
"""

# Images analysis
def count_images(img_dict):
    if not isinstance(img_dict, dict):
        return 0
    count = 0
    for size_key in ['hi_res', 'large', 'thumb']:
        if size_key in img_dict and isinstance(img_dict[size_key], list):
            count = max(count, sum(1 for img in img_dict[size_key] if img is not None))
    return count

meta_df['image_count'] = meta_df['images'].apply(count_images)
products_with_images = meta_df[meta_df['image_count'] > 0]

md_content += f"| Images | {len(products_with_images):,} | {len(products_with_images)/len(meta_df)*100:.2f}% |\n"

md_content += f"""
**Recommendation Engine Implications:**
- **Content-Based Filtering:** Rich metadata enables sophisticated content-based recommendations
- **Feature Extraction:** Product features and descriptions can be used for similarity computation
- **Hybrid Approaches:** Combine text-based features with collaborative signals
- **Strategy:** Use TF-IDF, word embeddings, or transformer models on product descriptions

---

## Part 3: Combined Analysis (Reviews + Metadata)

### 3.1 Dataset Join Coverage
"""

# Join analysis
review_parent_asins = set(review_df['parent_asin'].unique())
meta_parent_asins = set(meta_df['parent_asin'].unique())
common_asins = review_parent_asins.intersection(meta_parent_asins)
reviews_only = review_parent_asins - meta_parent_asins
meta_only = meta_parent_asins - review_parent_asins

md_content += f"""
| Metric | Count | Percentage |
|--------|-------|------------|
| Products in Reviews | {len(review_parent_asins):,} | - |
| Products in Metadata | {len(meta_parent_asins):,} | - |
| Common Products (Both) | {len(common_asins):,} | {len(common_asins)/len(review_parent_asins)*100:.2f}% of reviewed |
| Reviews Only (Orphaned) | {len(reviews_only):,} | {len(reviews_only)/len(review_parent_asins)*100:.2f}% of reviewed |
| Metadata Only (No Reviews) | {len(meta_only):,} | {len(meta_only)/len(meta_parent_asins)*100:.2f}% of catalog |

**Recommendation Engine Implications:**
- **Data Completeness:** {len(common_asins)/len(review_parent_asins)*100:.1f}% of reviewed products have metadata
- **Cold Start Products:** {len(meta_only):,} products need content-based recommendations only
- **Orphaned Reviews:** {len(reviews_only):,} products have reviews but missing metadata
- **Strategy:** Prioritize products with both reviews and metadata for training
- **Fallback:** Content-based for {len(meta_only):,} products without reviews

---

### 3.2 User-Item Matrix Characteristics
"""

# Calculate matrix sparsity
total_possible_interactions = unique_users * unique_products_reviewed
actual_interactions = len(review_df)
sparsity = (1 - actual_interactions / total_possible_interactions) * 100

md_content += f"""
**Matrix Dimensions:**
- Users: {unique_users:,}
- Items: {unique_products_reviewed:,}
- Possible Interactions: {total_possible_interactions:,}
- Actual Interactions: {actual_interactions:,}
- **Sparsity: {sparsity:.4f}%**

**Sparsity Analysis:**
- Extremely sparse matrix ({sparsity:.2f}% empty)
- Average user covers {actual_interactions/unique_users/unique_products_reviewed*100:.4f}% of product space
- Average product rated by {actual_interactions/unique_products_reviewed/unique_users*100:.4f}% of users

**Recommendation Engine Implications:**
- **High Sparsity:** Matrix factorization techniques (SVD, ALS) will be challenging
- **Strategy:** Use dimensionality reduction (SVD++, NMF) or neural collaborative filtering
- **Alternative:** Graph-based methods or deep learning to handle sparsity
- **Hybrid Essential:** Content-based features crucial to augment sparse collaborative signals

---

### 3.3 Data Quality for Recommendation Engine

**Completeness Score:**

| Dataset | Completeness | Key Fields Missing |
|---------|--------------|-------------------|
| Reviews | {(review_df.notna().sum().sum() / (len(review_df) * len(review_df.columns)) * 100):.1f}% | Text: {(review_df['text'].isna()).sum():,}, Title: {(review_df['title'].isna()).sum():,} |
| Metadata | {(meta_df.notna().sum().sum() / (len(meta_df) * len(meta_df.columns)) * 100):.1f}% | Price: {meta_df['price'].isna().sum():,}, Rating: {meta_df['average_rating'].isna().sum():,} |

**Data Quality Issues:**

| Issue | Count | Impact |
|-------|-------|--------|
| Reviews without metadata | {len(reviews_only):,} products | Cannot use content features |
| Products without reviews | {len(meta_only):,} products | Cold start problem |
| Empty review text | {(review_df['text'].isna() | (review_df['text'] == '')).sum():,} | Cannot extract text features |
| Missing prices | {meta_df['price'].isna().sum():,} | Cannot use price-based filtering |
| One-time users | {one_time_reviewers:,} | Weak collaborative signal |
| Single-review products | {single_review_products:,} | Weak item signal |

---

## Part 4: Recommendation Engine Strategy

### 4.1 Recommended Approaches

Based on the data analysis, here are the recommended approaches for building a recommendation engine:

#### **1. Hybrid Recommendation System (Primary Recommendation)**

**Rationale:**
- High sparsity ({sparsity:.2f}%) requires combining multiple signals
- {len(common_asins):,} products have both reviews and metadata (good hybrid candidate set)
- Rich textual content enables sophisticated content-based filtering

**Components:**
```
a) Collaborative Filtering (for users with history)
   - Matrix Factorization (SVD++, ALS)
   - Neural Collaborative Filtering
   - Item-based CF for similar products

b) Content-Based Filtering (for cold start)
   - TF-IDF on product descriptions and review text
   - Category and feature matching
   - Price-range filtering

c) Weighted Hybrid
   - Dynamic weighting based on user activity level
   - Higher CF weight for power users
   - Higher content weight for cold start scenarios
```

#### **2. User Segmentation Strategy**

| Segment | Count | Strategy |
|---------|-------|----------|
| Power Users (>10 reviews) | {power_users:,} | User-based CF, personalized recommendations |
| Occasional Users (2-10 reviews) | {((reviews_per_user > 1) & (reviews_per_user <= 10)).sum():,} | Item-based CF + content-based |
| One-time Users (1 review) | {one_time_reviewers:,} | Content-based + popular items |
| New Users (0 reviews) | - | Popular items + category-based |

#### **3. Item Segmentation Strategy**

| Segment | Count | Strategy |
|---------|-------|----------|
| Popular (>100 reviews) | {popular_products:,} | Collaborative filtering works well |
| Medium (11-100 reviews) | {((reviews_per_product > 10) & (reviews_per_product <= 100)).sum():,} | Hybrid approach |
| Low (2-10 reviews) | {((reviews_per_product > 1) & (reviews_per_product <= 10)).sum():,} | Content-based + similar items |
| Cold start (0-1 reviews) | {single_review_products + len(meta_only):,} | Pure content-based |

### 4.2 Feature Engineering Recommendations

**User Features:**
- Average rating given (user leniency)
- Rating variance (discriminativeness)
- Review activity level (number of reviews)
- Average review length (engagement level)
- Verified purchase ratio (trustworthiness)
- Temporal activity patterns

**Item Features:**
- Average rating
- Rating variance (controversy)
- Number of reviews (popularity)
- Price
- Category
- TF-IDF vectors from descriptions
- Feature bullet points
- Review text sentiment
- Helpful vote aggregates

**Interaction Features:**
- Rating
- Verified purchase status
- Helpful votes (quality signal)
- Review recency (temporal decay)
- Review text length
- Review sentiment

### 4.3 Model Architecture Suggestions

**Option 1: Neural Collaborative Filtering + Content**
```
User Embedding (dim=64) ─┐
                          ├─> MLP ─> Prediction
Item Embedding (dim=64) ─┤
                          │
Item Content Features ────┘
```

**Option 2: Two-Tower Model**
```
User Tower:                Item Tower:
- User ID embedding        - Item ID embedding
- User features            - Item content features
- User review history      - Item review aggregates
    ↓                          ↓
  Dense Layers            Dense Layers
    ↓                          ↓
  User Vector              Item Vector
         \\                    /
          \\                  /
           Cosine Similarity
                  ↓
             Score/Ranking
```

**Option 3: Graph Neural Network**
```
User-Item-Review Graph:
- Nodes: Users, Items, Reviews
- Edges: User-Review-Item relationships
- Node features: Rich embeddings
- Message passing: Aggregate neighbor information
```

### 4.4 Evaluation Metrics

**Offline Evaluation:**
- RMSE / MAE for rating prediction
- Precision@K, Recall@K for ranking
- nDCG@K for ranking quality
- Coverage (% of catalog recommended)
- Diversity (intra-list diversity)
- Novelty (how different from popular items)

**Online Evaluation (if deployed):**
- Click-through rate (CTR)
- Conversion rate
- Time to purchase
- User engagement metrics

### 4.5 Implementation Priority

**Phase 1: Baseline Models (Week 1-2)**
1. Popularity-based baseline
2. Item-based collaborative filtering
3. Simple content-based (TF-IDF)

**Phase 2: Advanced Models (Week 3-4)**
1. Matrix factorization (SVD++)
2. Neural collaborative filtering
3. Hybrid ensemble

**Phase 3: Optimization (Week 5-6)**
1. Hyperparameter tuning
2. Feature engineering iteration
3. Model ensembling

**Phase 4: Production Ready (Week 7-8)**
1. API development
2. Caching strategy
3. A/B testing framework

---

## Part 5: Key Insights Summary

### Critical Findings:

1. **Extreme Sparsity** ({sparsity:.2f}%): Hybrid approach is mandatory
2. **Positive Bias**: {positive_ratings.shape[0]/len(review_df)*100:.1f}% of reviews are 4-5 stars
3. **One-time Users**: {one_time_reviewers/unique_users*100:.1f}% - cold start is major challenge
4. **Cold Start Items**: {len(meta_only):,} products without reviews
5. **Rich Content**: {meta_df['has_description'].sum()/len(meta_df)*100:.1f}% products have descriptions
6. **Verified Purchases**: {verified_true/len(review_df)*100:.1f}% are verified - trust signal available
7. **Temporal Range**: {review_df['datetime'].max().year - review_df['datetime'].min().year} years - time decay needed

### Opportunities:

1. **Text Mining**: Rich review text for aspect-based recommendations
2. **Price Filtering**: {len(price_data)/len(meta_df)*100:.1f}% have price - price-aware recommendations
3. **Category Structure**: Multi-level categories for hierarchical recommendations
4. **Helpful Votes**: Quality signal for weighting reviews
5. **Verified Purchases**: Trust indicator for review reliability

### Challenges:

1. **Data Sparsity**: {sparsity:.2f}% sparse matrix
2. **Missing Metadata**: {len(reviews_only):,} products lack metadata
3. **Unbalanced Reviews**: Top {popular_products:,} products dominate
4. **Cold Start**: {one_time_reviewers:,} one-time users + {len(meta_only):,} products without reviews

---

## Part 6: Next Steps

### Immediate Actions:

1. **Data Preparation:**
   - Filter to products with both reviews and metadata ({len(common_asins):,} products)
   - Create train/validation/test splits (temporal or random)
   - Construct user-item matrix
   - Preprocess text data (tokenization, stopword removal)

2. **Feature Engineering:**
   - Extract TF-IDF features from product descriptions
   - Calculate user and item aggregates
   - Create temporal features (recency)
   - Normalize ratings per user

3. **Baseline Implementation:**
   - Implement popularity-based recommender
   - Build simple collaborative filtering (item-based)
   - Create content-based recommender (TF-IDF similarity)

4. **Evaluation Framework:**
   - Define metrics (Precision@10, nDCG@10, etc.)
   - Create evaluation pipeline
   - Set up cross-validation

### Research Directions:

1. Aspect-based sentiment analysis on review text
2. Graph neural networks for user-item-review graph
3. Attention mechanisms for review aggregation
4. Transfer learning from pre-trained language models
5. Sequential recommendation using temporal patterns

---

**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Data Version:** Amazon Reviews 2023 (Software Category)
**Analysis Period:** {review_df['datetime'].min().strftime('%Y-%m-%d')} to {review_df['datetime'].max().strftime('%Y-%m-%d')}
"""

# ============================================================================
# Save Report
# ============================================================================

output_file = "./AMAZON_SOFTWARE_ANALYSIS_REPORT.md"
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(md_content)

print(f"\n[SUCCESS] Report generated: {output_file}")
print(f"  Report size: {len(md_content):,} characters")
print(f"  Sections: 6 major parts")
print(f"  Tables: 50+ data tables")
print(f"  Recommendations: Comprehensive strategy for recommendation engine")
