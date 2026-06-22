

This subfolder (IBGE_method/own_LULC) is meant to generate a high-resolution Land-Use Land-Cover (LULC) for using at the IBGE method as replacement of the 


The exact algorithm, pre-processing steps and eventual post-processing are yet to be established


specifications:
- there are just two inputs, both of their paths are listed at IBGE_method/own_LULC/lulc_inputs.py:
    - the RBG orthophoto
    - the polygons of the study area, with classes encoded both as text labels and integer values
- in IBGE_method/own_LULC/lulc_inputs.py there are input parameters that can be changed, such as the output resolution of the LULC raster
- we shall investigate (plan) class imbalance and the need for data augmentation, as well as the best model architecture and hyperparameters
