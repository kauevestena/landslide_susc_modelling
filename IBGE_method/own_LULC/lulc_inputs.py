"""Configuration for the custom IBGE LULC workflow.

Keep all user-tunable hyperparameters in this file. Implementation modules
should read these values instead of defining local training/model defaults.
"""

input_polygons = "IBGE_method/own_LULC/extra_data/Classes_Uso_Solo.gpkg"
input_ortho = "/home/kaue/data/landslide/feb26/Ortho_4_GNSS-AAT_16cm.tif"


class_definitions = {
    1: {"name": "artif", "ibge_land_use_note": 10.0},
    2: {"name": "descob", "ibge_land_use_note": 5.0},
    3: {"name": "corpo_agua", "ibge_land_use_note": 0.0},
    4: {"name": "veg_campestre", "ibge_land_use_note": 2.0},
    5: {"name": "veg_florestal", "ibge_land_use_note": 1.0},
}


lulc_params = {
    "class_text_field": "Classe",
    "class_value_field": "Class_num",
    "output_resolution": 3.0,
    "output_folderpath": "IBGE_method/own_LULC/outputs",
    "output_nodata": 0,
    "ignore_index": 255,
    "lulc_filename": "lulc_custom_3m.tif",
    "probabilities_filename": "lulc_custom_probabilities_3m.tif",
    "metrics_filename": "lulc_training_metrics.json",
    "class_distribution_filename": "lulc_class_distribution.json",
    "metadata_filename": "lulc_model_metadata.json",
    "model_filename": "best_lulc_model.pth",
    "resampled_rgb_filename": "resampled_rgb_3m.tif",
    "training_labels_filename": "training_labels_3m.tif",
    "rasterize_all_touched": False,
    "use_alpha_valid_mask": True,
    "tile_size": 64,
    "stride": 64,
    "spatial_block_size": 1,
    "min_labeled_pixels_per_tile": 1,
    "train_percentage": 0.5,
    "validation_percentage": 0.35,
    "test_percentage": 0.15,
    "random_seed": 42,
    "model": {
        "architecture": "unet",
        "encoder": "resnet18",
        "encoder_weights": "imagenet",
        "input_channels": 3,
        "output_classes": 5,
    },
    "training": {
        "epochs": 100,
        "batch_size": 8,
        "learning_rate": 0.0003,
        "weight_decay": 0.0001,
        "optimizer": "adamw",
        "scheduler": "cosine",
        "cross_entropy_weight": 1.0,
        "dice_weight": 0.5,
        "class_weight_strategy": "inverse_frequency",
        "early_stopping_patience": 100,
        "num_workers": 0,
        "use_cuda": True,
        "save_best_metric": "macro_iou",
    },
    "augmentation": {
        "flip_probability": 0.5,
        "rotate90": True,
        "brightness": 0.08,
        "contrast": 0.12,
        "blur_enabled": False,
        "blur_kernel_size": 3,
        "noise_enabled": False,
        "noise_std": 0.01,
    },
    "inference": {
        "window_size": 128,
        "overlap": 32,
        "batch_size": 8,
        "use_cuda": True,
    },
    "postprocessing": {
        "smoothing_enabled": False,
        "smoothing_sigma": 0.75,
        "minimum_mapping_confidence": 0.0,
    },
    "smoke_test": {
        "epochs": 1,
        "max_train_tiles": 6,
        "max_eval_tiles": 2,
    },
}
