# Descrição completa do método

Este texto documenta a geração do produto IBGE adaptado de suscetibilidade a movimentos de massa. Ele se baseia na revisão metodológica de `IBGE_method/IBGE_method_full_review.md` e nos números efetivamente gerados nesta execução.

## Diferença entre o produto IBGE nacional e esta adaptação

O método nacional do IBGE integra seis temas ambientais em uma escala comum de potencialidade de 1 a 10: Geologia, Geomorfologia, Pedologia, Cobertura e uso da terra com Vegetação, Declividade e Pluviosidade. No fluxo nacional estrito, esses temas são agregados à Grade Estatística de 1 km x 1 km por máxima sobreposição.

Nesta adaptação, a grade de 1 km foi substituída pelo grid do DTM final do drone em 16 cm. A álgebra foi aplicada pixel a pixel, preservando pesos, sentido físico das notas e classes finais da metodologia IBGE.

O grid de referência foi `/home/kaue/data/landslide/dtm_final.tif`, em `EPSG:31984`, com 5.678 colunas por 18.227 linhas. A resolução é 0,16 m por 0,16 m, ou 0,0256 m² por pixel. O retângulo total tem 103.492.906 pixels.

## Cadeia operacional executada

- O DTM final foi usado como raster de referência para CRS, transform, shape e resolução.
- O processamento foi feito em tiles de 1024 pixels.
- A declividade foi calculada com halo de 1 pixel para reduzir artefatos de borda.
- Camadas categóricas foram alinhadas por vizinho mais próximo.
- Camadas contínuas, como pluviosidade antes da classificação, foram alinhadas por interpolação bilinear.
- Pixels com nodata, água ou nota temática inválida foram removidos da máscara válida.

## Insumos usados

- **Declividade / DECL:** `/home/kaue/data/landslide/dtm_final.tif`.
- **Cobertura e uso / USOVEG:** `/home/kaue/landslide_susc_modelling/IBGE_method/own_LULC/outputs_fullres/lulc_custom_ensemble_16cm.tif`.
- **Geologia / GEO:** `/home/kaue/Downloads/05_Geospatial_Data_Maps/sig_cachoeirodoitapemirim_es_suscet/data_interest_area/06.Geologia/Geologia.shp`, campo `SIGLA_UNID`.
- **Pedologia / PED:** `/home/kaue/Downloads/05_Geospatial_Data_Maps/sig_cachoeirodoitapemirim_es_suscet/data_interest_area/07.Pedologia/Pedologia.shp`, campo `DESC_`.
- **Geomorfologia / GEM:** `/home/kaue/Downloads/05_Geospatial_Data_Maps/sig_cachoeirodoitapemirim_es_suscet/data_interest_area/04.PadrõesDeRelevo/Relevo.shp`, campo `Classe`.
- **Pluviosidade / PLUV:** `/home/kaue/Downloads/05_Geospatial_Data_Maps/sig_cachoeirodoitapemirim_es_suscet/data_interest_area/08.Isoietas/AtlasPluviometrico/IsoietasAnuaisMedias/pma2`.

## Declividade, DECL

A declividade foi derivada do DTM final do drone e classificada pelos limiares IBGE: 0-3% = 1, 3-8% = 3, 8-20% = 5, 20-45% = 8, 45-75% = 9 e acima de 75% = 10. No produto, DECL teve 44.972.904 pixels avaliados, variou de 1,0000 a 10,0000, com média 6,4543 e desvio padrão 2,4883. Sua contribuição média ponderada foi 2,2590.

## Cobertura e uso da terra com vegetação, USOVEG

O LULC full-res por deep learning foi convertido para notas IBGE: área artificial = 10, área descoberta = 5, corpo d'água = 0 e exclusão, vegetação campestre = 2, vegetação florestal = 1. A distribuição no DTM válido foi: área artificial 14,06%, área descoberta 1,88%, corpo d'água 1,05%, vegetação campestre 64,56% e vegetação florestal 18,45%. A nota média USOVEG foi 2,9760.

## Geologia, GEO

No método estrito, GEO é média de litologia, gênese, província estrutural e subprovíncia estrutural. Nesta adaptação foi usado o proxy local pelo campo `SIGLA_UNID`. A unidade no recorte válido foi `PRps`, nota 7,3. A camada teve 44.972.904 pixels avaliados e ficou constante em 7,3000.

## Pedologia, PED

No método estrito, PED é `max(PROF, TEXT, RELTEXT)` para o solo dominante. Nesta adaptação foi usado o proxy pelo campo `DESC_`: `PEe1` = 8 e `Rios` = 0/excluído. A camada teve 44.420.881 pixels válidos, variando de 8,0000 a 8,0000.

## Geomorfologia, GEM

A geomorfologia usa padrões locais de relevo como proxy dos modelados BDIA. O mapeamento foi: Planícies e terraços fluviais = 1, Colinas = 3, Morros baixos = 7 e Morros altos = 9. GEM avaliou 44.972.904 pixels, variou de 1,0000 a 9,0000, com média 6,6379.

## Pluviosidade, PLUV

A pluviosidade veio do Atlas Pluviométrico CPRM/SGB PMA. Nesta adaptação, a nota foi interpolada continuamente entre limiares IBGE para preservar variação local. A PMA variou de 1225,1656 mm a 1235,3416 mm, gerando PLUV entre 6,9007 e 6,9414.

## Álgebra final

`S = 0,15*GEO + 0,20*GEM + 0,15*PED + 0,10*USOVEG + 0,35*DECL + 0,05*PLUV`

Contribuições médias ponderadas nesta execução:

- DECL: 2,2590.
- GEM: 1,3276.
- GEO: 1,0950.
- PED: 1,2000.
- USOVEG: 0,2976.
- PLUV: 0,3460.

## Resultado gerado

A máscara final contém 44.131.033 pixels válidos, equivalentes a 1.129.754,44 m², e 59.361.873 pixels inválidos, equivalentes a 1.519.663,95 m². A fração final válida foi 42,64%.

O score 1-10 variou de 3,2906 a 8,9420. A distribuição das cinco classes finais foi:

- Muito baixa: 337.031 pixels, 8.627,99 m², 0,76%.
- Baixa: 2.358.753 pixels, 60.384,08 m², 5,34%.
- Média: 4.826.588 pixels, 123.560,65 m², 10,94%.
- Alta: 14.759.546 pixels, 377.844,38 m², 33,44%.
- Muito alta: 21.849.115 pixels, 559.337,34 m², 49,51%.

A versão em três classes agregou o resultado em níveis gerais:

- Classe 1: 2.695.784 pixels, 69.012,07 m², 6,11%.
- Classe 2: 4.826.588 pixels, 123.560,65 m², 10,94%.
- Classe 3: 36.608.661 pixels, 937.181,72 m², 82,95%.

## Arquivos produzidos

- `ibge_class_map_3class.tif`.
- `ibge_class_map_5class.tif`.
- `ibge_land_use_custom_lulc.tif`.
- `ibge_note_geology.tif`.
- `ibge_note_geomorphology.tif`.
- `ibge_note_land_use_vegetation.tif`.
- `ibge_note_pedology.tif`.
- `ibge_note_pluviosity.tif`.
- `ibge_note_slope.tif`.
- `ibge_pluviosity_pma_mm.tif`.
- `ibge_susceptibility_score.tif`.
- `ibge_susceptibility_score_1to10.tif`.
- `ibge_valid_mask.tif`.

## Interpretação

O resultado deve ser interpretado como produto IBGE adaptado de alta resolução, adequado para inspeção local. Geologia, Pedologia e Geomorfologia são proxies locais compatíveis; eles preservam a escala e direção física da metodologia, mas não reproduzem integralmente as bases nacionais BDIA na grade de 1 km.
