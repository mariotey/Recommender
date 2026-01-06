import os
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

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

# Load directly from parquet files to avoid deprecated dataset script
review_dataset = load_dataset(
    "parquet",
    data_files=f"hf://datasets/{dataset_name}/raw_review_{category}/full/*.parquet",
    split="train"
)

print(f"Converting {len(review_dataset)} reviews to DataFrame...")
all_reviews = [review for review in tqdm(review_dataset, desc="Processing reviews")]
review_df = pd.DataFrame(all_reviews)

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

# Load directly from parquet files to avoid deprecated dataset script
meta_dataset = load_dataset(
    "parquet",
    data_files=f"hf://datasets/{dataset_name}/raw_meta_{category}/full/*.parquet",
    split="train"
)

print(f"Converting {len(meta_dataset)} metadata entries to DataFrame...")
meta_df = pd.DataFrame(meta_dataset)

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
print(f"  Reviews: {review_df.shape[0]:,} rows {review_df.shape[1]} columns")
print(f"  Metadata: {meta_df.shape[0]:,} rows {meta_df.shape[1]} columns")
