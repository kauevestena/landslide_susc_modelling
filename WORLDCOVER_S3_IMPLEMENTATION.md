# ESA WorldCover S3 Direct Download Implementation

## Summary

Implemented a cleaner and more accurate method for fetching ESA WorldCover land cover data by downloading raw classification GeoTIFFs directly from AWS S3, replacing the previous WMS-based approach that required color-to-class mapping.

## Problem

The initial implementation used ESA WorldCover's WMS service, which returns **rendered RGB/RGBA visualization tiles** instead of raw classification data. Converting these visualization colors back to class codes introduced significant errors:

### WMS Color-Mapping Issues (Observed)
- **Wrong class distribution**: An area that is primarily forest/grassland was classified as:
  - 47% "built-up" (urban)
  - 37% "moss/lichen"
  - Only 0.32% tree cover (should be ~43%)
- **Approximate palette mapping**: WMS rendering may use slightly different colors than expected
- **Nearest-color fallbacks**: Ambiguous colors led to misclassification
- **Compression artifacts**: JPEG/PNG compression in visualization tiles introduced color variations

## Solution

Download raw classification GeoTIFFs directly from the public AWS S3 bucket maintained by ESA:
- **Bucket**: `s3://esa-worldcover` (no authentication required)
- **Tile format**: 3×3 degree tiles in EPSG:4326
- **Data format**: Single-band uint8 GeoTIFFs with direct class codes (10, 20, 30, ...)
- **Access**: Public HTTP or AWS CLI (`--no-sign-request`)

### Implementation Details

1. **Tile Discovery** (`get_worldcover_tile_names`)
   - Calculate which 3×3° tile(s) intersect the study area bbox
   - Handle areas spanning multiple tiles
   - Example tile naming: `ESA_WorldCover_10m_2021_v200_S21W042_Map.tif`

2. **Download** (`download_s3_tile`)
   - Try AWS CLI first (faster): `aws s3 cp s3://esa-worldcover/v200/2021/map/<tile> <output> --no-sign-request`
   - Fallback to HTTPS if AWS CLI unavailable
   - Cache tiles locally to avoid repeated downloads

3. **Mosaic** (`download_and_mosaic_s3_tiles`)
   - Download all intersecting tiles
   - Merge into single raster using `rasterio.merge`
   - Handle single-tile case efficiently (just copy)

4. **Integration**
   - Reproject to reference DTM grid (existing workflow)
   - One-hot encode classes (existing workflow)
   - Default method in `main_pipeline.py`

### Results Comparison

**Test Area**: Training DTM (`DTM_train.tif`)

| Metric | S3 Direct (Raw) | WMS (Color-Mapped) |
|--------|----------------|-------------------|
| **Tree Cover (%)** | 42.86% ✓ | 0.32% ✗ |
| **Grassland (%)** | 53.98% ✓ | 0.30% ✗ |
| **Built-up (%)** | 2.20% ✓ | 47.04% ✗ |
| **Moss/Lichen (%)** | Not present | 37.19% ✗ |
| **Data Quality** | Clean class codes | Color mapping errors |
| **Classes Detected** | 8 (correct) | 11 (incorrect) |

The S3 method produces **accurate** land cover classification matching the expected forest/grassland landscape, while the WMS method misclassified the area as primarily urban and moss-covered.

## Usage

### Standalone Script

```bash
# S3 direct download (recommended)
python src/external_data_fetch/fetch_esa_worldcover.py \
    /path/to/dtm.tif \
    artifacts/lulc_cache \
    2021 \
    s3

# WMS fallback (if S3 unavailable)
python src/external_data_fetch/fetch_esa_worldcover.py \
    /path/to/dtm.tif \
    artifacts/lulc_cache \
    2021 \
    wms
```

### Pipeline Integration

The main pipeline now uses S3 by default:

```yaml
# config.yaml
preprocessing:
  external_lulc:
    enabled: true
    source: worldcover
    worldcover:
      year: 2021  # 2020 or 2021
```

No configuration changes needed—S3 download is the default method.

### Programmatic Usage

```python
from src.external_data_fetch import fetch_and_process_worldcover

# S3 direct (recommended)
one_hot, raw, classes = fetch_and_process_worldcover(
    reference_raster="dtm.tif",
    output_dir="lulc_cache",
    year=2021,
    use_s3=True  # Default
)

# WMS fallback
one_hot, raw, classes = fetch_and_process_worldcover(
    reference_raster="dtm.tif",
    output_dir="lulc_cache",
    year=2021,
    use_s3=False
)
```

## Requirements

### S3 Direct Method (Preferred)
- **AWS CLI** (optional, for faster downloads): `pip install awscli` or `apt install awscli`
- **No AWS credentials needed**: Uses `--no-sign-request` for public access
- **Fallback to HTTPS**: If AWS CLI unavailable, downloads via HTTPS automatically

### WMS Method (Fallback)
- No additional requirements
- Less accurate (color mapping)
- Smaller downloads (rendered tiles vs full-resolution COGs)

## Benefits

✅ **Accuracy**: Direct class codes, no conversion errors  
✅ **Reliability**: Official ESA data, no WMS rendering variations  
✅ **Performance**: Cloud-Optimized GeoTIFFs for efficient partial reads  
✅ **Caching**: Individual tiles cached, reused across multiple runs  
✅ **No Auth**: Public S3 bucket, no registration or API keys  
✅ **Fallback**: WMS method still available if S3 fails  

## Limitations

- **Download size**: Raw tiles are larger than WMS rendered images (10-50 MB per 3×3° tile)
- **Internet required**: First-time download requires internet; subsequent runs use cache
- **AWS CLI recommended**: HTTPS fallback works but is slower for large tiles

## Files Modified

- `src/external_data_fetch/fetch_esa_worldcover.py`
  - Added S3 tile discovery and download functions
  - Added mosaic support for multi-tile areas
  - Kept WMS method as fallback
  - Updated CLI to support method selection

- `src/main_pipeline.py`
  - Updated to use `use_s3=True` by default
  - No config changes needed

## References

- **ESA WorldCover Data Access**: https://esa-worldcover.org/en/data-access
- **AWS Open Data Registry**: https://registry.opendata.aws/esa-worldcover/
- **S3 Bucket**: s3://esa-worldcover
- **Product User Manual**: https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/docs/WorldCover_PUM_V2.0.pdf

## Testing

```bash
# Test with your training area
source .venv/bin/activate
python src/external_data_fetch/fetch_esa_worldcover.py \
    /path/to/your/DTM_train.tif \
    artifacts/test_lulc \
    2021 \
    s3

# Verify output
python -c "
import rasterio
import numpy as np
with rasterio.open('artifacts/test_lulc/worldcover_2021_reprojected.tif') as src:
    data = src.read(1)
    unique = sorted(np.unique(data[data > 0]))
    print(f'Classes found: {unique}')
    for val in unique:
        pct = 100 * np.sum(data == val) / data.size
        print(f'  Class {val}: {pct:.2f}%')
"
```

## Migration Notes

Existing cached WMS data in `artifacts/derived/*/lulc_cache/` will remain. To regenerate with S3:

```bash
# Option 1: Delete cache and rerun
rm -rf artifacts/derived/*/lulc_cache/
python -m src.main_pipeline

# Option 2: Force recreate
python -m src.main_pipeline --force_recreate
```

The pipeline will automatically detect missing files and download fresh S3 data.
