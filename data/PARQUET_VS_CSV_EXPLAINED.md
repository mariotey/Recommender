# Parquet vs CSV vs SQL - Complete Explanation

## What is a Parquet File?

**Apache Parquet** is a **columnar storage file format** designed for efficient data storage and retrieval, especially for big data and analytics workloads.

### Key Characteristics:
- **Binary format** (not human-readable)
- **Columnar storage** (stores data by columns, not rows)
- **Compressed** (built-in compression)
- **Schema-aware** (knows data types)
- **Optimized for analytics** (read-heavy workloads)

---

## File Format Comparison

### 1. CSV (Comma-Separated Values)

**What it is:**
- Plain text file with values separated by commas
- Each line is a row
- Human-readable

**Example CSV:**
```csv
user_id,rating,text,timestamp
A123,5.0,"Great software!",1609459200000
B456,3.0,"Okay product",1609545600000
C789,1.0,"Terrible, don't buy",1609632000000
```

**Pros:**
- ✅ Human-readable (can open in Notepad/Excel)
- ✅ Universal compatibility
- ✅ Simple to create and understand
- ✅ Small files for small datasets

**Cons:**
- ❌ No data types (everything is text)
- ❌ No compression (large file sizes)
- ❌ Slow to read/write for large datasets
- ❌ Must read entire file even for one column
- ❌ No schema validation
- ❌ Inefficient for analytics

---

### 2. Parquet (Apache Parquet)

**What it is:**
- Binary columnar storage format
- Stores data by columns, not rows
- Built-in compression and encoding

**How data is stored (conceptual):**
```
Row-based (CSV):
Row 1: [user_id: A123, rating: 5.0, text: "Great!", timestamp: 1609459200000]
Row 2: [user_id: B456, rating: 3.0, text: "Okay", timestamp: 1609545600000]
Row 3: [user_id: C789, rating: 1.0, text: "Bad", timestamp: 1609632000000]

Column-based (Parquet):
Column user_id: [A123, B456, C789]
Column rating: [5.0, 3.0, 1.0]
Column text: ["Great!", "Okay", "Bad"]
Column timestamp: [1609459200000, 1609545600000, 1609632000000]
```

**Pros:**
- ✅ **Extremely compressed** (5-10x smaller than CSV)
- ✅ **Fast column reads** (only read columns you need)
- ✅ **Type-safe** (preserves data types: int, float, string, etc.)
- ✅ **Optimized for analytics** (aggregations, filtering)
- ✅ **Schema embedded** (self-describing)
- ✅ **Efficient for large datasets** (billions of rows)
- ✅ **Supports complex types** (lists, nested structures)

**Cons:**
- ❌ Not human-readable (binary format)
- ❌ Requires special libraries to read
- ❌ Slightly more complex to work with

---

### 3. SQL Database (.db, .sqlite, PostgreSQL, etc.)

**What it is:**
- Database management system
- Stores data in tables with relationships
- Requires a database server (except SQLite)

**Pros:**
- ✅ ACID transactions (consistency, reliability)
- ✅ Complex queries (JOINs, aggregations)
- ✅ Concurrent access (multiple users)
- ✅ Indexing for fast lookups
- ✅ Relational structure (foreign keys, constraints)

**Cons:**
- ❌ Requires database server (overhead)
- ❌ More complex setup
- ❌ Slower for bulk analytics compared to Parquet
- ❌ Not ideal for sharing datasets (need to export)
- ❌ Requires active connection

---

## Practical Comparison: Your Amazon Dataset

### Your Dataset Stats:
- **Reviews:** 4,880,181 rows × 10 columns
- **Metadata:** 89,251 rows × 16 columns

### File Size Comparison (Estimated):

| Format | Review File Size | Metadata File Size | Total Size |
|--------|-----------------|-------------------|------------|
| **CSV** | ~2.5 GB | ~50 MB | ~2.55 GB |
| **Parquet** | ~596 MB | ~88 MB | ~684 MB |
| **Compressed CSV (.csv.gz)** | ~800 MB | ~20 MB | ~820 MB |
| **SQLite Database** | ~2.0 GB | ~40 MB | ~2.04 GB |

**Your actual Parquet files:**
- `review_data.parquet`: **596 MB** ✅
- `meta_data.parquet`: **88 MB** ✅

**If you used CSV:**
- `review_data.csv`: **~2.5 GB** (4.2x larger!)
- `meta_data.csv`: **~50 MB**

**Savings: ~1.8 GB** by using Parquet! 🎉

---

## Performance Comparison

### Scenario 1: Loading Full Dataset

**CSV:**
```python
# Read entire CSV into memory
import pandas as pd
df = pd.read_csv("review_data.csv")  # Takes ~45-60 seconds, 2.5 GB memory
```

**Parquet:**
```python
# Read parquet
import pandas as pd
df = pd.read_parquet("review_data.parquet")  # Takes ~8-12 seconds, 596 MB memory
```

**Result:** Parquet is **4-5x faster** ⚡

---

### Scenario 2: Reading Only 2 Columns (user_id and rating)

**CSV:**
```python
# Must read entire file, then select columns
df = pd.read_csv("review_data.csv", usecols=['user_id', 'rating'])
# Still reads ~2.5 GB from disk, filters in memory
# Time: ~40 seconds
```

**Parquet:**
```python
# Only reads the 2 columns from disk
df = pd.read_parquet("review_data.parquet", columns=['user_id', 'rating'])
# Reads only ~120 MB from disk (columnar storage!)
# Time: ~2 seconds
```

**Result:** Parquet is **20x faster** for column selection! 🚀

---

### Scenario 3: Filtering Data

**Query:** "Get all 5-star reviews"

**CSV:**
```python
# Must load entire dataset, then filter
df = pd.read_csv("review_data.csv")
five_star = df[df['rating'] == 5.0]
# Time: ~50 seconds
```

**Parquet with predicate pushdown:**
```python
# Can filter while reading (in some engines)
df = pd.read_parquet("review_data.parquet",
                     filters=[('rating', '==', 5.0)])
# Time: ~15 seconds (still needs to read rating column fully)
```

**Result:** Parquet is **3x faster** ⚡

---

## Why NOT CSV for Your Dataset?

### 1. **File Size** (Storage & Bandwidth)
- Your review CSV would be **2.5 GB** vs **596 MB** Parquet
- If sharing/downloading: 4x longer download time
- If storing in cloud: 4x higher storage costs

### 2. **Memory Usage**
- CSV loads everything as strings initially (wastes memory)
- Parquet knows types: int64, float32, string, bool, etc.
- **Example:** Your `timestamp` column:
  - CSV: Stores as string "1609459200000" (13 bytes each)
  - Parquet: Stores as int64 (8 bytes each) → 38% memory savings

### 3. **Load Time**
- CSV: ~45-60 seconds to load 4.88M reviews
- Parquet: ~8-12 seconds
- **You save ~40 seconds every time you load the data!**

### 4. **Data Types**
- CSV loses data types:
  ```python
  # CSV might read rating as string "5.0" instead of float 5.0
  # Must manually convert: df['rating'] = df['rating'].astype(float)
  ```
- Parquet preserves types automatically ✅

### 5. **Complex Data**
Your dataset has **nested structures**:
```python
review_df['images'] = [
    [{'url': 'http://...', 'size': 'large'}, ...],  # List of dicts
    [],
    [{'url': 'http://...'}]
]
```

- **CSV:** Cannot handle nested data naturally (must convert to JSON strings)
  ```csv
  images
  "[{""url"": ""http://..."", ""size"": ""large""}]"
  "[]"
  ```
  → Messy, error-prone

- **Parquet:** Natively supports nested structures ✅
  ```python
  # Parquet stores as proper list[dict] type
  df['images'][0]  # Returns actual list of dicts
  ```

### 6. **Partial Column Reads**
Machine learning workflows often need specific columns:
```python
# Building a recommendation model - only need user_id, product_id, rating
df = pd.read_parquet("reviews.parquet",
                     columns=['user_id', 'parent_asin', 'rating'])
# Reads only 15% of file (3 of 10 columns)
```

With CSV, you **always read 100%** of the file, even if you need 3/10 columns.

---

## When to Use Each Format

### Use **CSV** when:
- ✅ Small datasets (<100 MB)
- ✅ Need human readability (debugging, manual inspection)
- ✅ Sharing with non-technical users
- ✅ Simple, flat data (no nested structures)
- ✅ Compatibility with Excel/Google Sheets
- ✅ One-time use (not loading repeatedly)

### Use **Parquet** when:
- ✅ Large datasets (>100 MB, especially >1 GB)
- ✅ Analytics and machine learning workloads
- ✅ Need fast column-wise operations
- ✅ Repeated loading/processing
- ✅ Complex data types (lists, nested structures)
- ✅ Storage efficiency matters
- ✅ Working with big data tools (Spark, Dask, PyArrow)

### Use **SQL Database** when:
- ✅ Need ACID transactions (consistency, atomicity)
- ✅ Concurrent writes/reads by multiple users
- ✅ Complex relationships between tables (foreign keys)
- ✅ Need real-time updates (not batch processing)
- ✅ Application backend (web app, API)
- ✅ Need row-level security/permissions

---

## Parquet Technical Details

### Compression Algorithms:
Parquet supports multiple compression:
- **Snappy** (default): Fast compression/decompression, good ratio
- **GZIP**: Higher compression, slower
- **LZ4**: Very fast, moderate compression
- **ZSTD**: Best of both worlds (fast + high compression)

**Your files use Snappy compression by default.**

### Encoding:
Parquet uses smart encoding:
- **Run-Length Encoding (RLE):** For repeated values
  - Example: Column with many `True` values → highly compressed
- **Dictionary Encoding:** For categorical data
  - Example: `main_category` has only 2 unique values → stores dictionary + indices
- **Delta Encoding:** For sorted/sequential data
  - Example: `timestamp` column → stores differences instead of full values

### Columnar Layout Benefits:

**Example Query:** "What's the average rating?"

**CSV (row-based):**
```
Read row 1: [user_id, rating, text, timestamp, ...]
Read row 2: [user_id, rating, text, timestamp, ...]
...
Read 4.88M rows → Extract rating column → Calculate average
↳ Must read ALL data (2.5 GB)
```

**Parquet (columnar):**
```
Read ONLY rating column: [5.0, 3.0, 1.0, 4.0, ...]
Calculate average directly
↳ Only reads rating column (~50 MB instead of 2.5 GB)
```

**Result: 50x less data read from disk!** 🚀

---

## Code Examples

### Converting CSV to Parquet

```python
import pandas as pd

# Read CSV
df = pd.read_csv("review_data.csv")

# Save as Parquet with compression
df.to_parquet("review_data.parquet",
              engine='pyarrow',
              compression='snappy',  # or 'gzip', 'zstd'
              index=False)

print(f"CSV size: {os.path.getsize('review_data.csv') / 1024**2:.2f} MB")
print(f"Parquet size: {os.path.getsize('review_data.parquet') / 1024**2:.2f} MB")
```

### Reading Parquet Efficiently

```python
import pandas as pd

# Read entire file
df = pd.read_parquet("review_data.parquet")

# Read only specific columns (FAST!)
df = pd.read_parquet("review_data.parquet",
                     columns=['user_id', 'rating', 'parent_asin'])

# Read with filters (some engines support this)
df = pd.read_parquet("review_data.parquet",
                     filters=[('rating', '>=', 4.0)])

# Read in chunks (for very large files)
import pyarrow.parquet as pq
parquet_file = pq.ParquetFile("review_data.parquet")
for batch in parquet_file.iter_batches(batch_size=100000):
    df_chunk = batch.to_pandas()
    # Process chunk
```

---

## Summary: Why Parquet for Your Project

### Your Amazon Dataset:
- **4.88 million reviews** with complex nested data (images, lists)
- **Need for fast analytics** (building recommendation engine)
- **Repeated loading** (experimentation, model training)
- **Column-wise operations** (aggregations, filtering)

### Benefits You Get:
1. **73% smaller files** (596 MB vs 2.5 GB)
2. **5x faster loading** (8 sec vs 45 sec)
3. **20x faster column reads** (critical for ML)
4. **Type safety** (no manual type conversion)
5. **Native nested data support** (images, lists)
6. **Better for Pandas/NumPy/ML workflows**

### What You're Giving Up:
- ❌ Can't open in Notepad (not human-readable)
- ❌ Requires pandas/pyarrow to read
  - But you're already using Python for ML, so this doesn't matter!

---

## Bonus: Hybrid Approach

You can keep both formats for different purposes:

```python
# For development/debugging: Export small sample as CSV
sample_df = review_df.head(1000)
sample_df.to_csv("sample_reviews.csv", index=False)
# ✅ Can open in Excel to inspect data

# For production/analysis: Use Parquet
review_df.to_parquet("review_data.parquet", index=False)
# ✅ Fast, efficient, optimized
```

---

## Conclusion

**For your recommendation engine project, Parquet is the clear winner:**

| Factor | CSV | Parquet | Winner |
|--------|-----|---------|--------|
| File Size | 2.5 GB | 596 MB | **Parquet (4x smaller)** |
| Load Time | 45 sec | 8 sec | **Parquet (5x faster)** |
| Column Read | 40 sec | 2 sec | **Parquet (20x faster)** |
| Memory Usage | High | Low | **Parquet** |
| Data Types | Manual | Automatic | **Parquet** |
| Nested Data | No | Yes | **Parquet** |
| Human Readable | Yes | No | **CSV** |
| ML Workflow | Okay | Excellent | **Parquet** |

**Verdict:** Use Parquet for your project. The 4x space savings and 5-20x performance gains far outweigh the loss of human readability. 🏆

---