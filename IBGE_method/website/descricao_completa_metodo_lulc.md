# Descrição completa do método LULC

Esta aba documenta o método de geração da camada LULC usada como tema USOVEG no produto IBGE adaptado. O objetivo do LULC foi substituir uma base regional ou nacional de uso e cobertura da terra por um produto local treinado com polígonos interpretados sobre ortofoto de drone.

A implementação está versionada no repositório, principalmente em `IBGE_method/own_LULC/lulc_inputs.py` e nos módulos de `IBGE_method/own_LULC/implementation/`. O arquivo `lulc_inputs.py` é a superfície única de hiperparâmetros: entradas, classes, resolução, divisão espacial, modelos, perdas, treino, inferência e ensemble ficam declarados ali.

## Entradas e codificação de classes

- Polígonos de treinamento: `IBGE_method/own_LULC/extra_data/Classes_Uso_Solo.gpkg`.
- Ortofoto RGB/RGBA: `/home/kaue/data/landslide/feb26/Ortho_4_GNSS-AAT_16cm.tif`.
- Campo textual: `Classe`.
- Campo numérico: `Class_num`.
- Pixels ignorados no treinamento: `255`.
- Nodata no raster LULC final: `0`.

As classes usadas no raster final são os códigos originais 1 a 5. Internamente, durante o treinamento, elas são convertidas para índices 0 a 4 porque a função de perda multiclasse espera classes contíguas. Na escrita do GeoTIFF final, a codificação volta para 1 a 5.

- Classe 1: `artif`, nota IBGE USOVEG 10,0.
- Classe 2: `descob`, nota IBGE USOVEG 5,0.
- Classe 3: `corpo_agua`, nota IBGE USOVEG 0,0.
- Classe 4: `veg_campestre`, nota IBGE USOVEG 2,0.
- Classe 5: `veg_florestal`, nota IBGE USOVEG 1,0.

## Conjuntos de variáveis

Foram previstos dois conjuntos de atributos derivados apenas da ortofoto, sem insumos externos adicionais:

- `rgb`: red, green, blue.
- `rgb_indices`: red, green, blue, hue, saturation, value, brightness, excess_green, local_texture.

O conjunto `rgb_indices` amplia o RGB com HSV, brilho, excesso de verde e textura local. Ele foi incluído para ajudar a separar vegetação, solo exposto, água e superfícies artificiais quando a assinatura espectral RGB pura é ambígua.

## Divisão espacial e desenho de avaliação

- Estratégia de split: `stratified_spatial_blocks`.
- Percentuais por classe: treino 50,00%, validação 35,00%, teste 15,00%.
- Exigência de todas as classes em cada split: `True`.
- Tentativas máximas de seeds para achar split viável: `400`.

A divisão não é uma amostragem aleatória simples de pixels. Ela usa blocos espaciais para reduzir vazamento espacial entre treino, validação e teste. Além disso, as metas de proporção são aplicadas por classe: a ideia é que cada classe tenha suporte próprio em treino, validação e teste, evitando avaliações artificialmente boas ou ruins por ausência de uma classe minoritária.

## Amostragem balanceada e aumento de dados

- Sampler: `class_balanced_weighted`.
- Potência para classes raras: `1.0`.
- Peso máximo por tile: `20.0`.
- Otimizador: `adamw`.
- Scheduler: `cosine`.
- Taxa de aprendizado: `0.0003`.
- Decaimento de peso: `0.0001`.
- Batch size: `8`.
- Épocas: `100`.
- Early stopping patience: `100`.
- Aumentos: flips com probabilidade 0.5, rotações de 90 graus `True`, brilho 0.08, contraste 0.12.

O sampler pondera tiles com maior presença de classes raras. Isso aumenta a frequência com que água, solo exposto ou outras classes de menor área aparecem nos batches de treinamento, sem alterar a avaliação. A validação e o teste continuam feitos nos blocos espaciais mantidos fora do treino.

## Inferência full-resolution

- Janela de inferência: `256` pixels.
- Sobreposição: `64` pixels.
- Batch de inferência: `8`.
- CUDA habilitado quando disponível: `True`.

A inferência percorre a ortofoto em janelas sobrepostas. Para cada pixel, as probabilidades acumuladas nas janelas que o cobrem são médias antes da decisão final. O resultado individual de cada modelo inclui um raster de classes e um raster multibanda de probabilidades, com cinco bandas, uma por classe.

## Os cinco modelos full-res

- Modelo 1: U-Net++ / resnet18.
- Identificador: `fullres_unetplusplus_resnet18_focal_lovasz_rgb_indices_seed42`.
- Configuração: resolução 0,16 m; feature set `rgb_indices`; perda `focal_lovasz`; tile 192; stride 96; seed 42.
- Treinamento: 100 épocas planejadas/executadas; melhor época 61.
- Validação: macro IoU 0,9973, macro F1 0,9987, acurácia global 0,9996.
- Teste final: macro IoU 0,9767, macro F1 0,9880, acurácia global 0,9980.

- Modelo 2: DeepLabV3+ / resnet18.
- Identificador: `fullres_deeplabv3plus_resnet18_focal_dice_rgb_seed42`.
- Configuração: resolução 0,16 m; feature set `rgb`; perda `focal_dice`; tile 192; stride 96; seed 42.
- Treinamento: 100 épocas planejadas/executadas; melhor época 68.
- Validação: macro IoU 0,9978, macro F1 0,9989, acurácia global 0,9997.
- Teste final: macro IoU 0,9992, macro F1 0,9996, acurácia global 0,9996.

- Modelo 3: U-Net / resnet34.
- Identificador: `fullres_unet_resnet34_weighted_ce_lovasz_rgb_seed42`.
- Configuração: resolução 0,16 m; feature set `rgb`; perda `weighted_ce_lovasz`; tile 192; stride 96; seed 42.
- Treinamento: 100 épocas planejadas/executadas; melhor época 100.
- Validação: macro IoU 0,9992, macro F1 0,9996, acurácia global 0,9997.
- Teste final: macro IoU 0,9956, macro F1 0,9978, acurácia global 0,9974.

- Modelo 4: U-Net / mit_b0.
- Identificador: `fullres_unet_mit_b0_focal_lovasz_rgb_seed42`.
- Configuração: resolução 0,16 m; feature set `rgb`; perda `focal_lovasz`; tile 192; stride 96; seed 42.
- Treinamento: 100 épocas planejadas/executadas; melhor época 99.
- Validação: macro IoU 0,9908, macro F1 0,9953, acurácia global 0,9970.
- Teste final: macro IoU 0,9920, macro F1 0,9960, acurácia global 0,9990.

- Modelo 5: FPN / resnet34.
- Identificador: `fullres_fpn_resnet34_weighted_ce_lovasz_rgb_seed42`.
- Configuração: resolução 0,16 m; feature set `rgb`; perda `weighted_ce_lovasz`; tile 192; stride 96; seed 42.
- Treinamento: 100 épocas planejadas/executadas; melhor época 77.
- Validação: macro IoU 0,9983, macro F1 0,9991, acurácia global 0,9990.
- Teste final: macro IoU 0,9782, macro F1 0,9888, acurácia global 0,9972.

## Ensemble

- Tipo de seleção: `ensemble`.
- Estratégia: `probability_average`.
- Política de votantes: `diverse_top_models`.
- Número de votantes: `5`.
- Validação do ensemble: macro IoU 1,0000, macro F1 1,0000, acurácia global 1,0000, pixels avaliados 1.452.814.
- Teste final do ensemble: macro IoU 0,9972, macro F1 0,9986, acurácia global 0,9998, pixels avaliados 639.541.
- Concordância média entre votos duros: 0,4088.
- Confiança média da probabilidade média: 0,9612.
- Margem média entre as duas maiores probabilidades: 0,9319.
- Entropia média normalizada: 0,0736.

O ensemble usa média de probabilidades. Esse método é preferido ao voto majoritário simples porque preserva a confiança de cada modelo. Em cada pixel, as cinco distribuições de probabilidade são somadas e normalizadas; a classe final é a classe com maior probabilidade média. A concordância por voto duro é gravada como diagnóstico separado, junto com confiança, margem e entropia.

## Observação sobre placeholders

Métricas lidas dos artefatos locais em outputs_fullres/.
