# External LULC Data Fetchers

This module provides utilities to fetch and process external Land Use/Land Cover (LULC) datasets for landslide susceptibility modeling.

## Available Datasets

### 1. ESA WorldCover (10m)

**Source:** European Space Agency  
**Resolution:** 10 meters  
**Coverage:** Global  
**Years Available:** 2020, 2021  
**Access:** Free via WMS API (no authentication required)

**Land Cover Classes:**
- 10: Tree cover
- 20: Shrubland
- 30: Grassland
- 40: Cropland
- 50: Built-up
- 60: Bare / sparse vegetation
- 70: Snow and ice
- 80: Permanent water bodies
- 90: Herbaceous wetland
- 95: Mangroves
- 100: Moss and lichen

**Documentation:** https://esa-worldcover.org/

**Usage:**
```python
from src.external_data_fetch import fetch_and_process_worldcover

one_hot_path, raw_path, class_info = fetch_and_process_worldcover(
    reference_raster="/path/to/dtm.tif",
    output_dir="artifacts/derived/train/lulc_cache",
    year=2021,
    force_download=False,
)
```

**Command-line usage:**
```bash
python -m src.external_data_fetch.fetch_esa_worldcover \
    /path/to/dtm.tif \
    artifacts/lulc_cache \
    2021
```

---

### 2. Google Dynamic World (10m)

**Source:** Google Earth Engine  
**Resolution:** 10 meters  
**Coverage:** Global  
**Temporal:** Near real-time (daily updates)  
**Access:** Requires Google Earth Engine account and authentication

**Land Cover Classes:**
- 0: Water
- 1: Trees
- 2: Grass
- 3: Flooded vegetation
- 4: Crops
- 5: Shrub and scrub
- 6: Built
- 7: Bare
- 8: Snow and ice

**Features:**
- Provides both discrete classification and per-class probabilities
- Can composite over date ranges for temporal stability
- Ideal for capturing seasonal variations or recent land cover changes

**Documentation:** https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1

**Setup:**
```bash
# Install Earth Engine API
pip install earthengine-api

# Authenticate
earthengine authenticate

# Optional: Initialize with GCP project
earthengine --project=your-project-id authenticate
```

**Usage:**
```python
from src.external_data_fetch import fetch_and_process_dynamic_world

one_hot_path, raw_path, class_info = fetch_and_process_dynamic_world(
    reference_raster="/path/to/dtm.tif",
    output_dir="artifacts/derived/train/lulc_cache",
    start_date="2023-01-01",
    end_date="2023-06-30",
    use_probabilities=False,  # Use hard classification
    force_download=False,
    ee_project="your-gcp-project-id",  # Optional
)
```

**Command-line usage:**
```bash
python -m src.external_data_fetch.fetch_dynamic_world \
    /path/to/dtm.tif \
    artifacts/lulc_cache \
    2023-01-01 \
    2023-06-30 \
    your-gcp-project-id
```

---

## Integration with Pipeline

### Configuration (`config.yaml`)

```yaml
preprocessing:
  external_lulc:
    enabled: true  # Set to false to use K-means clustering instead
    source: worldcover  # Options: worldcover, dynamic_world
    
    # WorldCover-specific settings
    worldcover:
      year: 2021  # 2020 or 2021
    
    # Dynamic World-specific settings
    dynamic_world:
      start_date: null  # YYYY-MM-DD, null for auto (last 6 months)
      end_date: null
      use_probabilities: false  # Use probability bands vs hard classification
      ee_project: null  # GCP project ID for Earth Engine
    
    force_download: false  # Force re-download even if cached
```

### Automatic Integration

When `external_lulc.enabled: true` in `config.yaml`, the pipeline will:

1. **Download** the specified LULC dataset for each study area's extent
2. **Reproject** to match the DTM reference grid (resolution, projection, extent)
3. **One-hot encode** the land cover classes into binary channels
4. **Stack** LULC channels with topographic and spectral features
5. **Cache** downloaded data in `artifacts/derived/{area}/lulc_cache/` to avoid repeated downloads

### Feature Channels

With external LULC enabled, the feature stack becomes:
- **14 channels:** Topographic derivatives (elevation, slope, curvatures, flow, etc.)
- **3 channels:** RGB orthophoto (normalized)
- **N channels:** One-hot encoded LULC classes (N varies by dataset and area)

Example from WorldCover:
```
lulc_class_0: tree_cover
lulc_class_1: shrubland
lulc_class_2: grassland
lulc_class_3: cropland
lulc_class_4: built_up
lulc_class_5: bare_sparse_veg
```

---

## Comparison: K-means vs External LULC

### K-means Clustering (Default)
**Pros:**
- No external dependencies
- No internet required
- Works offline

**Cons:**
- Arbitrary spectral clusters (no semantic meaning)
- Not transferable between study areas
- Sensitive to illumination and seasonal variations
- Confuses spectrally similar but semantically different classes

### External LULC (WorldCover/Dynamic World)
**Pros:**
- Semantically meaningful classes
- Consistent across study areas
- Incorporates expert knowledge and validation
- Better generalization
- Validated against ground truth data

**Cons:**
- Requires internet connection (first time)
- Dynamic World requires Earth Engine authentication
- 10m resolution may not capture fine-scale features in high-res orthophotos
- Fixed class schema (cannot customize)

---

## Troubleshooting

### WorldCover

**Error: Download timeout**
- The WMS service may be slow for large areas
- Increase timeout in `fetch_esa_worldcover.py` (default: 120s)
- Try reducing the download resolution (adjust `width` and `height` parameters)

**Error: No data returned**
- Check if your area is within WorldCover coverage (global coverage, but check year availability)
- Verify bounding box coordinates are valid

### Dynamic World

**Error: Earth Engine not initialized**
```bash
earthengine authenticate
```

**Error: No project set**
- Provide `ee_project` in config or initialize with project:
```bash
earthengine --project=your-project-id authenticate
```

**Error: Export too large**
- Reduce date range for smaller composite
- Decrease max_pixels parameter
- Split area into smaller tiles

---

## Advanced Usage

### Custom Class Aggregation

You can aggregate classes into broader categories by modifying the class mapping:

```python
from src.external_data_fetch import one_hot_encode_worldcover

custom_mapping = {
    10: "vegetation",  # Tree cover
    20: "vegetation",  # Shrubland
    30: "vegetation",  # Grassland
    40: "agriculture",  # Cropland
    50: "urban",       # Built-up
    60: "bare",        # Bare/sparse veg
    # ... etc
}

one_hot, class_info = one_hot_encode_worldcover(
    "worldcover_reprojected.tif",
    "worldcover_aggregated.tif",
    class_mapping=custom_mapping,
)
```

### Temporal Compositing (Dynamic World)

For areas with cloud cover or seasonal changes, use longer date ranges:

```python
# 1-year composite for maximum stability
fetch_and_process_dynamic_world(
    reference_raster="dtm.tif",
    output_dir="lulc_cache",
    start_date="2022-01-01",
    end_date="2022-12-31",
)
```

---

## Performance Notes

- **Caching:** Downloaded LULC data is cached per area. Subsequent pipeline runs reuse cached data unless `force_download: true`.
- **Download time:** WorldCover typically downloads in 10-60 seconds. Dynamic World may take 1-5 minutes depending on date range.
- **Disk space:** Expect ~5-20 MB per study area for cached LULC data.

---

## References

- ESA WorldCover: https://esa-worldcover.org/
- Google Dynamic World: https://www.dynamicworld.app/
- Earth Engine Python API: https://developers.google.com/earth-engine/tutorials/community/intro-to-python-api
