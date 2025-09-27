import os
import yaml
import numpy as np
import torch
import rasterio
from rasterio.windows import Window
import torch.nn as nn
import segmentation_models_pytorch as smp
from tqdm import tqdm

def run_inference(config):
    """Runs inference on the test set and generates output maps."""
    print("--- Starting Inference ---")

    # --- 1. Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    derived_data_dir = config['project_structure']['derived_data_dir']
    experiments_dir = config['project_structure']['experiments_dir']
    outputs_dir = 'outputs/'
    os.makedirs(outputs_dir, exist_ok=True)

    feature_stack_path = os.path.join(derived_data_dir, 'feature_stack.tif')
    model_path = os.path.join(experiments_dir, 'best_model.pth')

    # --- 2. Load Model ---
    model = smp.Unet(
        encoder_name=config['model']['encoder'],
        encoder_weights=None, # No need to load weights from imagenet
        in_channels=config['model']['in_channels'],
        classes=config['model']['out_classes'],
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # --- 3. Sliding Window Inference ---
    print("Performing sliding window inference...")
    window_size = config['inference']['window_size']
    overlap = config['inference']['overlap']
    stride = window_size - overlap

    with rasterio.open(feature_stack_path) as src:
        width = src.width
        height = src.height
        meta = src.meta.copy()

        preds = np.zeros((height, width), dtype=np.float32)
        counts = np.zeros((height, width), dtype=np.float32)

        for y in tqdm(range(0, height, stride)):
            for x in range(0, width, stride):
                h = min(window_size, height - y)
                w = min(window_size, width - x)

                window = Window(x, y, w, h)
                tile = src.read(window=window)
                tile = torch.from_numpy(tile).float().unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(tile).sigmoid().cpu().numpy().squeeze()

                preds[y:y+h, x:x+w] += output[:h, :w]
                counts[y:y+h, x:x+w] += 1

        # Average the predictions
        preds /= counts

    # --- 4. Save Susceptibility Map ---
    susceptibility_path = os.path.join(outputs_dir, 'susceptibility_map.tif')
    meta.update(dtype='float32', count=1)
    with rasterio.open(susceptibility_path, 'w', **meta) as dst:
        dst.write(preds.astype(np.float32), 1)
    print(f"Susceptibility map saved to {susceptibility_path}")

    # --- 5. Uncertainty Map (Monte Carlo Dropout) ---
    print("Generating uncertainty map with Monte Carlo dropout...")
    mc_iterations = config['inference']['mc_dropout_iterations']

    # Enable dropout during inference
    def enable_dropout(m):
        if type(m) == nn.Dropout:
            m.train()

    model.apply(enable_dropout)

    mc_preds = []
    for _ in tqdm(range(mc_iterations)):
        with rasterio.open(feature_stack_path) as src:
            with torch.no_grad():
                preds = np.zeros((height, width), dtype=np.float32)
                counts = np.zeros((height, width), dtype=np.float32)

                for y in range(0, height, stride):
                    for x in range(0, width, stride):
                        h = min(window_size, height - y)
                        w = min(window_size, width - x)

                        window = Window(x, y, w, h)
                        tile = src.read(window=window)
                        tile = torch.from_numpy(tile).float().unsqueeze(0).to(device)

                        output = model(tile).sigmoid().cpu().numpy().squeeze()

                        preds[y:y+h, x:x+w] += output[:h, :w]
                        counts[y:y+h, x:x+w] += 1

                preds /= counts
                mc_preds.append(preds)

    uncertainty = np.std(mc_preds, axis=0)

    # --- 6. Save Uncertainty Map ---
    uncertainty_path = os.path.join(outputs_dir, 'uncertainty_map.tif')
    with rasterio.open(uncertainty_path, 'w', **meta) as dst:
        dst.write(uncertainty.astype(np.float32), 1)
    print(f"Uncertainty map saved to {uncertainty_path}")

    print("--- Inference Complete ---")

if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    run_inference(config)
