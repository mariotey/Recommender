# Amazon Software Dataset - Data Dictionary

**Dataset:** Amazon Reviews 2023 - Software Category
**Last Updated:** 2026-01-06
**Total Records:** 4,880,181 reviews | 89,251 products
**Date Range:** 1999-03-15 to 2023-09-11

---

## Table of Contents
1. [Overview](#overview)
2. [Data Model](#data-model)
3. [Review Data Schema](#review-data-schema)
4. [Metadata Schema](#metadata-schema)
5. [Data Relationships](#data-relationships)
6. [Data Quality Notes](#data-quality-notes)

---

## Overview

The dataset consists of two main files:

| File | Records | Description |
|------|---------|-------------|
| `review_data.parquet` | 4,880,181 | User reviews and ratings for software products |
| `meta_data.parquet` | 89,251 | Product metadata and catalog information |

**Primary Key Relationship:** Both datasets are linked via `parent_asin` field.

---

## Data Model

```
┌─────────────────────────┐         ┌──────────────────────────┐
│   REVIEW_DATA           │         │      META_DATA           │
├─────────────────────────┤         ├──────────────────────────┤
│ • rating                │         │ • main_category          │
│ • title                 │         │ • title                  │
│ • text                  │         │ • average_rating         │
│ • images[]              │         │ • rating_number          │
│ • asin                  │         │ • features[]             │
│ • parent_asin (FK) ────────────── │ • parent_asin (PK)       │
│ • user_id               │         │ • description[]          │
│ • timestamp             │         │ • price                  │
│ • helpful_vote          │         │ • images[]               │
│ • verified_purchase     │         │ • videos[]               │
└─────────────────────────┘         │ • store                  │
                                    │ • categories[]           │
                                    │ • details{}              │
                                    │ • bought_together        │
                                    │ • subtitle               │
                                    │ • author                 │
                                    └──────────────────────────┘

Cardinality: Many-to-One (Many reviews → One product)
```

---

## Review Data Schema

**File:** `review_data.parquet`
**Rows:** 4,880,181
**Columns:** 10

### Column Specifications

| Column | Data Type | Nullable | Description | Example | Statistics |
|--------|-----------|----------|-------------|---------|------------|
| `rating` | `float64` | No | Star rating given by user (1.0-5.0) | `1.0` | Mean: 3.94, Median: 5.0 |
| `title` | `string` | No | Review title/headline | `"malware"` | Avg length: 16.9 chars |
| `text` | `string` | No | Full review text content | `"mcaffee IS malware"` | Avg length: 140.1 chars, 174 empty |
| `images` | `array<string>` | No | Array of image URLs attached to review | `[]` | Most reviews have no images |
| `asin` | `string` | No | Amazon Standard Identification Number (product variant) | `"B07BFS3G7P"` | Unique product identifier |
| `parent_asin` | `string` | No | Parent ASIN (product family identifier, foreign key) | `"B0BQSK9QCF"` | Links to meta_data |
| `user_id` | `string` | No | Anonymized user identifier | `"AGCI7FAH4GL5FI6..."` | 2,589,466 unique users |
| `timestamp` | `int64` | No | Unix timestamp in milliseconds | `1562182632076` | Range: 921470... to 169439... |
| `helpful_vote` | `int64` | No | Number of helpful votes received | `0` | Mean: 4.92, Max: 10,267 |
| `verified_purchase` | `boolean` | No | Whether review is from verified purchase | `False` | 95.19% are verified |

### Review Data Details

#### Rating Distribution
```
Rating 1.0: 695,854 reviews (14.26%)
Rating 2.0: 239,253 reviews (4.90%)
Rating 3.0: 419,356 reviews (8.59%)
Rating 4.0: 857,082 reviews (17.56%)
Rating 5.0: 2,668,636 reviews (54.68%)
```

#### Text Characteristics
- **Title:** Average 3.2 words, median 11 characters
- **Text:** Average 26.4 words, median 85 characters
- **Empty Reviews:** 174 reviews (0.00%)
- **Pattern:** Negative reviews (1-2 stars) tend to be longer (avg 192-219 chars)

#### Timestamp Format
- **Type:** Unix epoch in milliseconds
- **Example:** `1562182632076` → 2019-07-03 19:17:12 UTC
- **Conversion:** `datetime.fromtimestamp(timestamp / 1000)`

#### Helpful Votes
- **Range:** -1 to 10,267
- **Distribution:** 57.44% have 0 votes, 42.56% have >0 votes
- **Negative Values:** Rare, may indicate data issues

---

## Metadata Schema

**File:** `meta_data.parquet`
**Rows:** 89,251
**Columns:** 16

### Column Specifications

| Column | Data Type | Nullable | Description | Example | Missing Count |
|--------|-----------|----------|-------------|---------|---------------|
| `parent_asin` | `string` | No | Product family identifier (primary key) | `"B00VRPSGEO"` | 0 |
| `main_category` | `string` | Yes | Primary product category | `"Appstore for Android"` | 1,769 (1.98%) |
| `title` | `string` | No | Product name/title | `"Accupressure Guide"` | 0 |
| `average_rating` | `float64` | Yes | Platform average rating (1.0-5.0) | `3.6` | 25 (0.03%) |
| `rating_number` | `float64` | Yes | Total number of ratings | `NaN` | 2,959 (3.32%) |
| `features` | `array<string>` | No | Product feature bullet points | `["All the pressing point..."]` | 0 (but can be empty array) |
| `description` | `array<string>` | No | Product description paragraphs | `["Acupressure technique..."]` | 0 (but can be empty array) |
| `price` | `float64` | Yes | Product price in USD | `0.0` | 18,266 (20.47%) |
| `images` | `array<object>` | No | Array of image objects with URLs | `[{hi_res, large, thumb, variant}]` | 0 (but can be empty) |
| `videos` | `array<object>` | No | Array of video objects | `[{title, url, user_id}]` | 0 (but can be empty) |
| `store` | `string` | Yes | Seller/developer name | `"mAppsguru"` | 213 (0.24%) |
| `categories` | `array<string>` | No | Hierarchical category path | `[]` | 0 (but can be empty) |
| `details` | `object` | No | Dictionary of product specifications | `{key: value}` | 0 |
| `bought_together` | `unknown` | Yes | Frequently bought together items | `None` | 89,251 (100%) |
| `subtitle` | `float64` | Yes | Product subtitle | `NaN` | 89,251 (100%) |
| `author` | `float64` | Yes | Product author/creator | `NaN` | 89,251 (100%) |

### Metadata Details

#### Main Category Distribution
```
Appstore for Android: 68,679 (76.95%)
Software: 18,791 (21.05%)
Gift Cards: 4 (0.00%)
Other categories: <10 products each
```

#### Price Distribution
```
Free (price = 0.0): 54,778 products (61.38%)
$0-10: 13,305 products (82.09% of paid)
$10-25: 1,194 products (7.37% of paid)
$25-50: 924 products (5.70% of paid)
$50+: 784 products (4.84% of paid)
Max price: $1,998.00
```

#### Images Array Structure
Each image object contains:
```json
{
  "hi_res": "URL or null",
  "large": "URL",
  "thumb": "URL or null",
  "variant": "MAIN|PT01|PT02|..."
}
```

#### Videos Array Structure
Each video object contains:
```json
{
  "title": "string",
  "url": "string",
  "user_id": "string"
}
```

#### Details Object Structure
The `details` field is a dictionary with ~70 possible keys including:
- **Product Info:** `Manufacturer`, `Brand`, `Model Name`, `Item model number`
- **App-Specific:** `Minimum Operating System`, `Version`, `Size`, `Date first listed on Amazon`
- **Technical:** `Application Permissions`, `Approximate Download Time`
- **Physical (if applicable):** `Package Dimensions`, `Item Weight`
- **Other:** `Release Date`, `Language`, `Developed By`, `Best Sellers Rank`

**Note:** Most keys have `None` values. Only relevant fields are populated per product.

---

## Data Relationships

### Primary Relationship
```
review_data.parent_asin → meta_data.parent_asin
```

### Join Coverage
- **Products in reviews:** 89,246 unique parent_asin values
- **Products in metadata:** 89,251 unique parent_asin values
- **Common products:** 89,246 (100% of reviewed products)
- **Orphaned reviews:** 0 reviews without metadata
- **Cold start products:** 5 products without any reviews

### User-Product Interaction Matrix
- **Users:** 2,589,466 unique users
- **Products:** 89,246 unique products
- **Interactions:** 4,880,181 reviews
- **Sparsity:** 99.9979% (extremely sparse)

---

## Data Quality Notes

### Completeness

| Dataset | Field Completeness | Critical Missing Fields |
|---------|-------------------|------------------------|
| Reviews | 100% | None (all required fields present) |
| Metadata | 83.7% | Price (20.47%), rating_number (3.32%) |

### Known Issues

1. **Empty Text Reviews:** 174 reviews have empty text field (0.00%)
2. **Negative Helpful Votes:** Some reviews have -1 helpful votes (data anomaly)
3. **Unused Metadata Fields:** `bought_together`, `subtitle`, `author` are 100% null
4. **Missing Content:** Some products have empty `features` or `description` arrays
5. **Timestamp Range:** Data spans 24 years (1999-2023), with 78% from 2012-2023

### Data Type Notes

1. **Arrays vs Nulls:** Empty arrays `[]` are different from `None`/`NaN`
2. **Timestamp:** Stored as int64 milliseconds, not datetime
3. **Boolean:** `verified_purchase` uses numpy bool type
4. **Nested Objects:** `images`, `videos`, and `details` contain complex nested structures

### Recommendations for Usage

1. **Join Strategy:** Use `parent_asin` for joins between datasets
2. **Price Handling:** 61% of products are free, handle price=0.0 appropriately
3. **Text Preprocessing:** Review text may contain special characters, URLs, HTML
4. **Timestamp Conversion:** Divide by 1000 and use datetime functions
5. **Array Fields:** Check array length before accessing elements
6. **Missing Data:** Handle NaN/None values in price, rating_number, store
7. **Sparsity:** Consider hybrid recommendation approaches due to 99.99% sparsity

---

## Usage Examples

### Loading Data
```python
import pandas as pd

# Load datasets
reviews = pd.read_parquet('data/output/review_data.parquet')
metadata = pd.read_parquet('data/output/meta_data.parquet')

# Join datasets
full_data = reviews.merge(
    metadata,
    on='parent_asin',
    how='left',
    suffixes=('_review', '_meta')
)
```

### Converting Timestamp
```python
from datetime import datetime

# Convert timestamp to datetime
reviews['review_date'] = pd.to_datetime(
    reviews['timestamp'],
    unit='ms'
)
```

### Handling Arrays
```python
# Check if features exist
metadata['has_features'] = metadata['features'].apply(lambda x: len(x) > 0)

# Extract first image URL
metadata['main_image'] = metadata['images'].apply(
    lambda x: x[0]['large'] if len(x) > 0 else None
)
```

### Filtering Verified Reviews
```python
# Get only verified purchase reviews
verified_reviews = reviews[reviews['verified_purchase'] == True]
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-06 | Initial data dictionary creation |

---

**For Analysis Report:** See `AMAZON_SOFTWARE_ANALYSIS_REPORT.md` for detailed EDA and insights.
