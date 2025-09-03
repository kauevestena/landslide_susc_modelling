import os
import yaml
import numpy as np
import rasterio
from rasterio.transform import from_origin
import noise
from PySAGA_cmd import SAGA
import xdem
from sklearn.model_selection import train_test_split
import json
import shutil
from src.train import train_model
from src.inference import run_inference

def generate_dummy_data(config):
    """Generates dummy data."""
    print("--- Generating Dummy Data ---")
    raw_data_dir = config['project_structure']['raw_data_dir']
    dtm_shape = tuple(config['data_generation']['dtm_shape'])
    ortho_shape = tuple(config['data_generation']['ortho_shape'])
    gt_shape = tuple(config['data_generation']['gt_shape'])
    seed = config['reproducibility']['seed']

    os.makedirs(raw_data_dir, exist_ok=True)
    np.random.seed(seed)

    scale = 100.0
    octaves = 6
    persistence = 0.5
    lacunarity = 2.0
    dtm_data = np.zeros(dtm_shape)
    for i in range(dtm_shape[0]):
        for j in range(dtm_shape[1]):
            dtm_data[i][j] = noise.pnoise2(i / scale, j / scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity, repeatx=dtm_shape[0], repeaty=dtm_shape[1], base=seed)
    dtm_data = (dtm_data - np.min(dtm_data)) / (np.max(dtm_data) - np.min(dtm_data)) * 1000

    transform = from_origin(10, 50, 1, 1)
    crs = 'EPSG:32633'
    dtm_path = os.path.join(raw_data_dir, 'dtm.tif')
    with rasterio.open(dtm_path, 'w', driver='GTiff', height=dtm_data.shape[0], width=dtm_data.shape[1], count=1, dtype=dtm_data.dtype, crs=crs, transform=transform) as dst:
        dst.write(dtm_data, 1)

    ortho_data = np.random.randint(0, 256, size=ortho_shape, dtype=np.uint8)
    ortho_path = os.path.join(raw_data_dir, 'ortho.tif')
    with rasterio.open(ortho_path, 'w', driver='GTiff', height=ortho_data.shape[0], width=ortho_data.shape[1], count=3, dtype=ortho_data.dtype, crs=crs, transform=transform) as dst:
        dst.write(ortho_data[:, :, 0], 1)
        dst.write(ortho_data[:, :, 1], 2)
        dst.write(ortho_data[:, :, 2], 3)

    gt_data = np.zeros(gt_shape, dtype=np.uint8)
    for _ in range(10):
        x, y = np.random.randint(0, gt_shape[0] - 50), np.random.randint(0, gt_shape[1] - 50)
        size = np.random.randint(20, 50)
        class_label = np.random.randint(1, 4)
        gt_data[x:x+size, y:y+size] = class_label
    gt_path = os.path.join(raw_data_dir, 'ground_truth.tif')
    with rasterio.open(gt_path, 'w', driver='GTiff', height=gt_data.shape[0], width=gt_data.shape[1], count=1, dtype=gt_data.dtype, crs=crs, transform=transform) as dst:
        dst.write(gt_data, 1)
    print("--- Dummy Data Generation Complete ---")

def preprocess_data(config):
    """Preprocesses the raw data and generates a feature stack."""
    print("--- Starting Preprocessing ---")
    raw_data_dir = config['project_structure']['raw_data_dir']
    derived_data_dir = config['project_structure']['derived_data_dir']
    os.makedirs(derived_data_dir, exist_ok=True)

    dtm_path = os.path.join(raw_data_dir, 'dtm.tif')
    ortho_path = os.path.join(raw_data_dir, 'ortho.tif')

    saga = SAGA(saga_cmd=config['project_structure']['saga_cmd_path'])

    dtm_sdat_path = os.path.join(raw_data_dir, 'dtm.sdat')
    os.system(f"gdal_translate -of SAGA {dtm_path} {dtm_sdat_path}")
    dtm_filled_sdat_path = os.path.join(derived_data_dir, 'dtm_filled.sdat')
    saga_tool = saga / 'ta_preprocessor' / 'Sink Removal'
    saga_tool.execute(DEM=dtm_sdat_path, DEM_PREPROC=dtm_filled_sdat_path)

    dtm_filled_path = os.path.join(derived_data_dir, 'dtm_filled.tif')
    os.system(f"gdal_translate -of GTiff {dtm_filled_sdat_path} {dtm_filled_path}")

    dem = xdem.DEM(dtm_filled_path)

    def save_raster(data, path, crs, transform):
        with rasterio.open(path, 'w', driver='GTiff', height=data.shape[0], width=data.shape[1], count=1, dtype=data.dtype, crs=crs, transform=transform) as dst:
            dst.write(data, 1)

    slope_path = os.path.join(derived_data_dir, 'slope.tif')
    save_raster(dem.slope().data, slope_path, dem.crs, dem.transform)
    aspect_path = os.path.join(derived_data_dir, 'aspect.tif')
    save_raster(dem.aspect().data, aspect_path, dem.crs, dem.transform)
    planform_curvature_path = os.path.join(derived_data_dir, 'planform_curvature.tif')
    save_raster(dem.planform_curvature().data, planform_curvature_path, dem.crs, dem.transform)
    profile_curvature_path = os.path.join(derived_data_dir, 'profile_curvature.tif')
    save_raster(dem.profile_curvature().data, profile_curvature_path, dem.crs, dem.transform)
    tpi_path = os.path.join(derived_data_dir, 'tpi.tif')
    save_raster(dem.topographic_position_index().data, tpi_path, dem.crs, dem.transform)
    tri_path = os.path.join(derived_data_dir, 'tri.tif')
    save_raster(dem.terrain_ruggedness_index().data, tri_path, dem.crs, dem.transform)

    flow_accumulation_sdat_path = os.path.join(derived_data_dir, 'flow_accumulation.sdat')
    saga_tool = saga / 'ta_hydrology' / 'Flow Accumulation (Top-Down)'
    saga_tool.execute(ELEVATION=dtm_filled_sdat_path, FLOW=flow_accumulation_sdat_path, METHOD=4)
    flow_accumulation_path = os.path.join(derived_data_dir, 'flow_accumulation.tif')
    os.system(f"gdal_translate -of GTiff {flow_accumulation_sdat_path} {flow_accumulation_path}")

    twi_sdat_path = os.path.join(derived_data_dir, 'twi.sdat')
    saga_tool = saga / 'ta_hydrology' / 'SAGA Wetness Index'
    saga_tool.execute(DEM=dtm_filled_sdat_path, TWI=twi_sdat_path)
    twi_path = os.path.join(derived_data_dir, 'twi.tif')
    os.system(f"gdal_translate -of GTiff {twi_sdat_path} {twi_path}")

    with rasterio.open(flow_accumulation_path) as src:
        flow_accum_data = src.read(1)
        transform = src.transform
        crs = src.crs
    with rasterio.open(slope_path) as src:
        slope_data = src.read(1)
    cell_size = transform[0]
    sca_data = flow_accum_data * cell_size
    spi_data = sca_data * np.tan(np.deg2rad(slope_data))
    spi_path = os.path.join(derived_data_dir, 'spi.tif')
    save_raster(spi_data, spi_path, crs, transform)
    sti_data = (sca_data / 22.13)**0.6 * (np.sin(np.deg2rad(slope_data)) / 0.0896)**1.3
    sti_path = os.path.join(derived_data_dir, 'sti.tif')
    save_raster(sti_data, sti_path, crs, transform)

    ortho_norm_path = os.path.join(derived_data_dir, 'ortho_normalized.tif')
    shutil.copy(ortho_path, ortho_norm_path)
    with rasterio.open(ortho_path) as src:
        ortho_data = src.read()
        meta = src.meta
    lulc_data = np.random.randint(1, 7, size=(ortho_data.shape[1], ortho_data.shape[2]), dtype=np.uint8)
    lulc_path = os.path.join(derived_data_dir, 'lulc.tif')
    meta.update(count=1, dtype='uint8')
    with rasterio.open(lulc_path, 'w', **meta) as dst:
        dst.write(lulc_data, 1)

    feature_files = [slope_path, aspect_path, planform_curvature_path, profile_curvature_path, tpi_path, tri_path, flow_accumulation_path, twi_path, spi_path, sti_path, ortho_norm_path, lulc_path]
    stacked_features = []
    total_bands = 0
    for f in feature_files:
        with rasterio.open(f) as src:
            stacked_features.append(src.read())
            total_bands += src.count
    stacked_features = np.vstack(stacked_features)
    mean = stacked_features.mean(axis=(1, 2), keepdims=True)
    std = stacked_features.std(axis=(1, 2), keepdims=True)
    normalized_features = (stacked_features - mean) / std
    feature_stack_path = os.path.join(derived_data_dir, 'feature_stack.tif')
    meta.update(count=total_bands, dtype='float32')
    with rasterio.open(feature_stack_path, 'w', **meta) as dst:
        dst.write(normalized_features.astype(np.float32))
    print("--- Preprocessing Complete ---")

def prepare_dataset(config):
    """Prepares the dataset for model training and evaluation."""
    print("--- Starting Dataset Preparation ---")
    derived_data_dir = config['project_structure']['derived_data_dir']
    raw_data_dir = config['project_structure']['raw_data_dir']
    tiles_dir = config['project_structure']['tiles_dir']
    labels_dir = config['project_structure']['labels_dir']
    splits_dir = config['project_structure']['splits_dir']

    os.makedirs(tiles_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(splits_dir, exist_ok=True)

    feature_stack_path = os.path.join(derived_data_dir, 'feature_stack.tif')
    gt_path = os.path.join(raw_data_dir, 'ground_truth.tif')

    tile_size = config['dataset']['tile_size']
    tile_overlap = config['dataset']['tile_overlap']

    with rasterio.open(feature_stack_path) as src:
        features = src.read()
    with rasterio.open(gt_path) as src:
        ground_truth = src.read(1)

    height, width = features.shape[1:]
    n_blocks_x = 4
    n_blocks_y = 4
    block_width = width // n_blocks_x
    block_height = height // n_blocks_y
    blocks = []
    for i in range(n_blocks_y):
        for j in range(n_blocks_x):
            blocks.append((i * block_height, j * block_width))

    train_blocks, test_blocks = train_test_split(blocks, test_size=config['dataset']['test_size'], random_state=config['reproducibility']['seed'])
    train_blocks, val_blocks = train_test_split(train_blocks, test_size=config['dataset']['val_size'], random_state=config['reproducibility']['seed'])
    splits = {'train': train_blocks, 'val': val_blocks, 'test': test_blocks}

    stride = tile_size - tile_overlap
    for split_name, block_list in splits.items():
        split_tiles_dir = os.path.join(tiles_dir, split_name)
        split_labels_dir = os.path.join(labels_dir, split_name)
        os.makedirs(split_tiles_dir, exist_ok=True)
        os.makedirs(split_labels_dir, exist_ok=True)

        for block_y_start, block_x_start in block_list:
            for y in range(block_y_start, block_y_start + block_height - tile_size + 1, stride):
                for x in range(block_x_start, block_x_start + block_width - tile_size + 1, stride):
                    if y + tile_size <= height and x + tile_size <= width:
                        tile_features = features[:, y:y+tile_size, x:x+tile_size]
                        tile_gt = ground_truth[y:y+tile_size, x:x+tile_size]

                        if np.any(tile_features) and tile_features.shape[1:] == (tile_size, tile_size):
                            tile_name = f'tile_{y}_{x}.npy'
                            np.save(os.path.join(split_tiles_dir, tile_name), tile_features)
                            np.save(os.path.join(split_labels_dir, tile_name), tile_gt)

    split_info_path = os.path.join(splits_dir, 'splits.json')
    with open(split_info_path, 'w') as f:
        json.dump(splits, f, indent=4)
    print("--- Dataset Preparation Complete ---")

def main():
    """Main function to run the pipeline."""
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    generate_dummy_data(config)
    preprocess_data(config)
    prepare_dataset(config)
    train_model(config)
    run_inference(config)

if __name__ == '__main__':
    main()
