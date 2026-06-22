input_polygons = "IBGE_method/own_LULC/extra_data/Classes_Uso_Solo.gpkg"

input_ortho = "feb26/Ortho_4_GNSS-AAT_16cm.tif"


lulc_params = {
    "output_resolution": 10,  # in meters
    "output_folderpath": "IBGE_method/own_LULC/outputs",
    "train_percentage": 0.6,  # percentage of the data to be used for training
    "validation_percentage": 0.3,  # percentage of the data to be used for validation
    "test_percentage": 0.1,  # percentage of the data to be used for testing
}
