# Experimento de sensibilidade ao DTM

Este experimento compara o produto IBGE final com uma versão alternativa gerada a partir do DTM `DTM_OTJC_3_1_16cm.tif`, descrito como afetado por erros sistemáticos. O produto oficial não foi regenerado; ele foi usado apenas como referência de comparação.

A diferença dos scores foi calculada como `score alternativo - score final`. A diferença de elevação foi calculada como `DTM alternativo - DTM final`.

Valores positivos indicam aumento no produto derivado do DTM alternativo. Valores negativos indicam redução.

## Entradas

- DTM final: `/home/kaue/data/landslide/dtm_final.tif`.
- DTM alternativo: `/home/kaue/data/landslide/DTM_OTJC_3_1_16cm.tif`.
- LULC mantido: `/home/kaue/landslide_susc_modelling/IBGE_method/own_LULC/outputs_fullres/lulc_custom_ensemble_16cm.tif`.
- Score final: `/home/kaue/landslide_susc_modelling/IBGE_method/outputs/ibge_susceptibility_score_1to10.tif`.
- Score alternativo: `/home/kaue/landslide_susc_modelling/IBGE_method/dtm_sensitivity/DTM_OTJC_3_1_16cm/outputs/ibge_susceptibility_score_1to10.tif`.

## Pixels comparados

- Pixels válidos no produto final: 44.131.033.
- Pixels válidos no produto alternativo: 44.131.033.
- Pixels válidos comuns usados na comparação: 44.131.033.
- Área comum comparada: 1.129.754,44 m².
- Pixels válidos somente no produto final: 0.
- Pixels válidos somente no produto alternativo: 0.
- Pixels com DTM válido comum: 44.972.904.

## Estatísticas da diferença de score

- Média: -0,0105.
- Mediana: 0,0000.
- Desvio padrão: 0,5886.
- Mínimo: -3,1500.
- Máximo: 3,1500.
- P05: -1,0500.
- P25: 0,0000.
- P75: 0,0000.
- P95: 1,0500.
- Erro absoluto médio: 0,3471.
- RMSE: 0,5887.
- Fração com aumento do score: 21,26%.
- Fração com redução do score: 22,16%.

## Estatísticas da diferença entre DTMs

- Média: -0,2519 m.
- Mediana: -0,1776 m.
- Desvio padrão: 0,7016 m.
- Mínimo: -14,5500 m.
- Máximo: 13,0153 m.
- P05: -0,5517 m.
- P25: -0,2058 m.
- P75: -0,1516 m.
- P95: -0,0195 m.
- Erro absoluto médio: 0,3009 m.
- RMSE: 0,7454 m.
- Fração com DTM alternativo mais alto: 4,54%.
- Fração com DTM alternativo mais baixo: 95,46%.

## Mudança de classe final

- Mesma classe 5 níveis: 32.857.235 pixels, 74,45%.
- Classe aumentou no produto alternativo: 5.480.417 pixels, 12,42%.
- Classe diminuiu no produto alternativo: 5.793.381 pixels, 13,13%.

## Artefatos

- Raster de diferença de score: `/home/kaue/landslide_susc_modelling/IBGE_method/dtm_sensitivity/DTM_OTJC_3_1_16cm/comparison/score_difference_alt_minus_final.tif`.
- Raster de diferença entre DTMs: `/home/kaue/landslide_susc_modelling/IBGE_method/dtm_sensitivity/DTM_OTJC_3_1_16cm/comparison/dtm_difference_alt_minus_final.tif`.
- Prévia da diferença entre DTMs: `/home/kaue/landslide_susc_modelling/IBGE_method/dtm_sensitivity/DTM_OTJC_3_1_16cm/comparison/dtm_difference_gray.png`.
- Máscara de comparação: `/home/kaue/landslide_susc_modelling/IBGE_method/dtm_sensitivity/DTM_OTJC_3_1_16cm/comparison/valid_comparison_mask.tif`.
- Histograma PNG: `/home/kaue/landslide_susc_modelling/IBGE_method/dtm_sensitivity/DTM_OTJC_3_1_16cm/comparison/difference_histogram.png`.
- Histograma CSV: `/home/kaue/landslide_susc_modelling/IBGE_method/dtm_sensitivity/DTM_OTJC_3_1_16cm/comparison/difference_histogram.csv`.
- Estatísticas CSV da diferença entre DTMs: `/home/kaue/landslide_susc_modelling/IBGE_method/dtm_sensitivity/DTM_OTJC_3_1_16cm/comparison/dtm_difference_stats.csv`.
- Histograma CSV da diferença entre DTMs: `/home/kaue/landslide_susc_modelling/IBGE_method/dtm_sensitivity/DTM_OTJC_3_1_16cm/comparison/dtm_difference_histogram.csv`.
- Matriz de transição de classes: `/home/kaue/landslide_susc_modelling/IBGE_method/dtm_sensitivity/DTM_OTJC_3_1_16cm/comparison/class5_transition_matrix.csv`.
