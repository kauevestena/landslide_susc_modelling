# IBGE High-Resolution Adapted Compliance Report

This product keeps the IBGE thematic weights, note scale and final class breaks, but deliberately replaces the national 1 km statistical grid with the 16 cm drone DTM grid. Rainfall keeps the IBGE threshold anchors but is scored continuously between them so the high-resolution adapted product does not discard real PMA variation inside a broad national class.

## Grid and Algebra
- Compliance mode: `adapted_high_resolution_16cm`
- Adapted grid: `Drone DTM reference grid at 16 cm`
- Algebra: `Pixel-wise weighted map algebra on the 16 cm reference grid`

## Theme Inputs
- `DECL`: Drone DTM; `/home/kaue/data/landslide/dtm_final.tif`; Higher-resolution substitute for SRTM/CGIAR-CSI 90 m.
- `USOVEG`: Custom polygon-trained LULC; `/home/kaue/landslide_susc_modelling/IBGE_method/own_LULC/outputs_fullres/experiments/fullres_unet_resnet34_weighted_ce_lovasz_rgb_seed42/lulc_custom_16cm.tif`
- `GEM`: Geomorphology proxy; `/home/kaue/Downloads/05_Geospatial_Data_Maps/sig_cachoeirodoitapemirim_es_suscet/data_interest_area/04.PadrõesDeRelevo/Relevo.shp`; Local higher-resolution relief-pattern proxy for IBGE BDIA geomorphology.
- `GEO`: Geology proxy; `/home/kaue/Downloads/05_Geospatial_Data_Maps/sig_cachoeirodoitapemirim_es_suscet/data_interest_area/06.Geologia/Geologia.shp`; Local higher-resolution SIG/SGB geology proxy for IBGE BDIA geology.
- `PED`: Pedology proxy; `/home/kaue/Downloads/05_Geospatial_Data_Maps/sig_cachoeirodoitapemirim_es_suscet/data_interest_area/07.Pedologia/Pedologia.shp`; Local higher-resolution pedology proxy for IBGE BDIA pedology.
- `PLUV`: CPRM/SGB Atlas Pluviometrico annual mean precipitation; `/home/kaue/Downloads/05_Geospatial_Data_Maps/sig_cachoeirodoitapemirim_es_suscet/data_interest_area/08.Isoietas/AtlasPluviometrico/IsoietasAnuaisMedias/pma2`; IBGE-compatible rainfall input, resampled bilinearly and converted to a continuous adapted note through the IBGE rainfall thresholds.

## Pixel Coverage
- `slope`: valid-note pixels=44972904; zero/excluded on valid DTM=0
- `land_use`: valid-note pixels=44463789; zero/excluded on valid DTM=509115
- `geomorphology`: valid-note pixels=44972904; zero/excluded on valid DTM=0
- `geology`: valid-note pixels=44972904; zero/excluded on valid DTM=0
- `pedology`: valid-note pixels=44420881; zero/excluded on valid DTM=552023
- `pluviosity`: valid-note pixels=44972904; zero/excluded on valid DTM=0

## Final Outputs
- Valid fraction: `0.425999`
- Score range 1-10: `3.290574073791504` to `8.942044258117676`

See `summary.json`, `method_config.json`, and `provenance.json` for full machine-readable metadata.
