"""Data loading utilities for the content-based filtering module."""

import logging
from typing import Any

import pandas as pd

from content_based_filtering.config import ITEM_PARQUET, META_PARQUET, USER_ITEM_PARQUET

logger = logging.getLogger(__name__)


def load_meta() -> pd.DataFrame:
    """Load and merge product metadata with item feature store.

    Returns:
        DataFrame with columns: parent_asin, item_title, main_category,
        description, features, is_free.

    Raises:
        FileNotFoundError: If either parquet file is missing.
        ValueError: If required columns are absent after loading.
    """
    for path in (META_PARQUET, ITEM_PARQUET):
        if not path.exists():
            raise FileNotFoundError(f"Required data file not found: {path}")

    logger.info("Loading metadata from %s", META_PARQUET)
    meta = pd.read_parquet(META_PARQUET)

    required_cols = {"parent_asin", "item_title", "description", "features"}
    missing = required_cols - set(meta.columns)
    if missing:
        raise ValueError(f"meta_data.parquet is missing columns: {missing}")

    logger.info("Loading item features from %s", ITEM_PARQUET)
    items = pd.read_parquet(ITEM_PARQUET, columns=["parent_asin", "is_free"])

    meta = meta.merge(items, on="parent_asin", how="left")
    meta["is_free"] = meta["is_free"].fillna(False).infer_objects(copy=False)

    logger.info("Loaded %d products", len(meta))
    return meta


def load_user_item() -> pd.DataFrame:
    """Load user-item interaction data.

    Returns:
        DataFrame with columns: user_id, parent_asin, review_rating,
        recency_weight.

    Raises:
        FileNotFoundError: If the interaction parquet file is missing.
        ValueError: If required columns are absent after loading.
    """
    if not USER_ITEM_PARQUET.exists():
        raise FileNotFoundError(
            f"Required data file not found: {USER_ITEM_PARQUET}"
        )

    logger.info("Loading user-item interactions from %s", USER_ITEM_PARQUET)
    cols = ["user_id", "parent_asin", "review_rating", "recency_weight"]
    df = pd.read_parquet(USER_ITEM_PARQUET, columns=cols)

    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(
            f"user-item-interaction.parquet is missing columns: {missing}"
        )

    logger.info("Loaded %d interactions", len(df))
    return df


def build_item_text(row: Any) -> str:
    """Concatenate item title, description, and features into a single string.

    Args:
        row: A pandas Series representing one row of the metadata DataFrame.

    Returns:
        A whitespace-joined string of all text fields.
    """
    parts = [str(row["item_title"] or "")]

    desc = row["description"]
    feats = row["features"]

    if isinstance(desc, list):
        parts += [str(d) for d in desc if d]
    if isinstance(feats, list):
        parts += [str(f) for f in feats if f]

    return " ".join(parts)
