import os
import pandas as pd
from tqdm import tqdm
from huggingface_hub import hf_hub_url, list_repo_files
import requests

# Retrieved from: https://amazon-reviews-2023.github.io/
# NOTE: Use 2023 dataset as it is larger, more descriptive, more granular and cleaner compared to
#       previous datasets.
dataset_name = "McAuley-Lab/Amazon-Reviews-2023"

# NOTE: There are 33 categories + Unknown. Software is a subset.
category = "Software"

print(f"Loading Amazon Reviews 2023 dataset for category: {category}")
print("=" * 80)

#######################################################################################################
# Load User Reviews
#######################################################################################################
print("\n[1/4] Loading user reviews...")
print("Data Fields for User Reviews:")
print("  - rating (float)             -->   Rating of the product (from 1.0 to 5.0)")
print("  - title (str)                -->   Title of the user review")
print("  - text (str)                 -->   Text body of the user review")
print("  - images (list)              -->   Images that users post after they have received the product")
print("  - asin (str)                 -->   ID of the product")
print("  - parent_asin (str)          -->   Parent ID of the product")
print("  - user_id (str)              -->   ID of the reviewer")
print("  - timestamp (int)            -->   Time of the review (unix time)")
print("  - verified_purchase (bool)   -->   User purchase verification")
print("  - helpful_vote (int)         -->   Helpful votes of the review")
print()

# Find review files - they are in JSONL format, not parquet
print("  Discovering review files...")
all_files = list_repo_files(dataset_name, repo_type="dataset")

# Reviews are stored as JSONL files in raw/review_categories/
review_jsonl_file = f"raw/review_categories/{category}.jsonl"

if review_jsonl_file not in all_files:
    # Try with underscore format
    review_jsonl_file = f"raw/review_categories/{category.replace(' ', '_')}.jsonl"

if review_jsonl_file not in all_files:
    available_review_files = [f for f in all_files if f.startswith('raw/review_categories/') and f.endswith('.jsonl')]
    available_categories = [f.split('/')[-1].replace('.jsonl', '') for f in available_review_files]
    print(f"  ERROR: Review file not found: {review_jsonl_file}")
    print(f"  Available review categories: {sorted(available_categories)[:20]}")
    raise ValueError(f"No review file found for category '{category}'")

print(f"  Found review file: {review_jsonl_file}")

# Load JSONL file from HuggingFace
url = hf_hub_url(dataset_name, review_jsonl_file, repo_type="dataset")
print(f"  Loading reviews from JSONL...")

# Download and read JSONL file
import json
import urllib.request

response = urllib.request.urlopen(url)
reviews = []
for line in tqdm(response, desc="Reading reviews"):
    if line.strip():
        reviews.append(json.loads(line.decode('utf-8')))

review_df = pd.DataFrame(reviews)

print(f"✓ Loaded {len(review_df)} reviews")
print(f"  Columns: {list(review_df.columns)}")
print(f"  Shape: {review_df.shape}")

#######################################################################################################
# Load Item Metadata
#######################################################################################################
print("\n[2/4] Loading item metadata...")
print("Data Fields for Item Metadata:")
print("  - main_category (str)        -->   Main category (i.e., domain) of the product")
print("  - title (str)                -->   Name of the product")
print("  - average_rating (float)     -->   Rating of the product shown on the product page")
print("  - rating_number (int)        -->   Number of ratings in the product")
print("  - features (list)            -->   Bullet-point format features of the product")
print("  - description (list)         -->   Description of the product")
print("  - price (float)              -->   Price in US dollars (at time of crawling)")
print("  - images (list)              -->   Images of the product")
print("  - videos (list)              -->   Videos of the product including title and url")
print("  - store (str)                -->   Store name of the product")
print("  - categories (list)          -->   Hierarchical categories of the product")
print("  - details (dict)             -->   Product details, including materials, brand, sizes, etc")
print("  - parent_asin (str)          -->   Parent ID of the product")
print("  - bought_together (list)     -->   Recommended bundles from the websites")
print()

# Find metadata files - check for both parquet and JSONL
print("  Discovering metadata files...")

# Try parquet first
meta_parquet_files = [f for f in all_files if f.startswith(f"raw_meta_{category}/") and f.endswith(".parquet")]

if meta_parquet_files:
    print(f"  Found {len(meta_parquet_files)} parquet file(s)")
    # Load all parquet files
    meta_dfs = []
    for file_path in tqdm(meta_parquet_files, desc="Loading metadata"):
        url = hf_hub_url(dataset_name, file_path, repo_type="dataset")
        df = pd.read_parquet(url)
        meta_dfs.append(df)
    meta_df = pd.concat(meta_dfs, ignore_index=True)
else:
    # Fall back to JSONL
    meta_jsonl_file = f"raw/meta_categories/meta_{category}.jsonl"

    if meta_jsonl_file not in all_files:
        meta_jsonl_file = f"raw/meta_categories/meta_{category.replace(' ', '_')}.jsonl"

    if meta_jsonl_file not in all_files:
        raise ValueError(f"No metadata files found for category '{category}'")

    print(f"  Found metadata file: {meta_jsonl_file}")
    url = hf_hub_url(dataset_name, meta_jsonl_file, repo_type="dataset")
    print(f"  Loading metadata from JSONL...")

    response = urllib.request.urlopen(url)
    metadata = []
    for line in tqdm(response, desc="Reading metadata"):
        if line.strip():
            metadata.append(json.loads(line.decode('utf-8')))

    meta_df = pd.DataFrame(metadata)

print(f"✓ Loaded {len(meta_df)} metadata entries")
print(f"  Columns: {list(meta_df.columns)}")
print(f"  Shape: {meta_df.shape}")

#######################################################################################################
# Export Dataset
#######################################################################################################
print("\n[3/4] Exporting datasets to parquet...")
output_dir = "./output"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Export DataFrames
review_output_path = f"{output_dir}/review_data.parquet"
meta_output_path = f"{output_dir}/meta_data.parquet"

print(f"  Saving reviews to: {review_output_path}")
review_df.to_parquet(review_output_path, index=False)

print(f"  Saving metadata to: {meta_output_path}")
meta_df.to_parquet(meta_output_path, index=False)

print("\n[4/4] Complete!")
print("=" * 80)
print(f"✓ Successfully exported {len(review_df)} reviews to {review_output_path}")
print(f"✓ Successfully exported {len(meta_df)} metadata entries to {meta_output_path}")
print()
print("Summary:")
print(f"  Reviews: {review_df.shape[0]:,} rows × {review_df.shape[1]} columns")
print(f"  Metadata: {meta_df.shape[0]:,} rows × {meta_df.shape[1]} columns")
