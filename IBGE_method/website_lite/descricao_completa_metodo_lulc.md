# Descrição completa do método LULC Lite

Esta versão lite do relatório IBGE usa uma única rede LULC individual. Ela não usa média de probabilidades, voto majoritário, concordância entre modelos ou qualquer outro componente de ensemble. O objetivo é documentar uma alternativa mais simples e mais leve: selecionar o melhor modelo individual full-resolution e usar diretamente o seu raster LULC como USOVEG no IBGE adaptado.

A página principal do projeto continua documentando o produto com ensemble. Esta página lite é paralela: seus outputs IBGE, thumbnails, webviewer, tabelas e textos são derivados de `outputs_lite/`, `reports_lite/` e `configs_lite/`. Isso evita misturar resultados do ensemble com resultados do modelo único.

## Critério de seleção

O critério de seleção é maior macro IoU de validação entre runs full-res, excluindo smoke runs e excluindo o próprio ensemble. Em caso de empate, a ordenação usa macro IoU de teste e depois `run_id`. O teste permanece como confirmação final; ele não é o critério principal porque isso contaminaria o holdout de teste com decisão de modelo.

Modelo selecionado: `fullres_unet_resnet34_weighted_ce_lovasz_rgb_seed42`.
Arquitetura selecionada: U-Net / resnet34.
Feature set: `rgb`.
Função de perda: `weighted_ce_lovasz`.
Tile/stride: 192 / 96.
Seed: 42.
Melhor época: 100.
Validação: macro IoU 0,9992, macro F1 0,9996, acurácia global 0,9997.
Teste final: macro IoU 0,9956, macro F1 0,9978, acurácia global 0,9974.

## Comparação dos candidatos

Os cinco candidatos full-res foram treinados com resolução de 16 cm, tiles de 192 pixels e stride de 96 pixels. A tabela HTML abaixo apresenta as mesmas métricas em formato tabular; a lista aqui resume a justificativa de seleção.

- 1. `fullres_unet_resnet34_weighted_ce_lovasz_rgb_seed42` (selecionado): U-Net / resnet34, features `rgb`, loss `weighted_ce_lovasz`, val macro IoU 0,9992, test macro IoU 0,9956.
- 2. `fullres_fpn_resnet34_weighted_ce_lovasz_rgb_seed42` (candidato): FPN / resnet34, features `rgb`, loss `weighted_ce_lovasz`, val macro IoU 0,9983, test macro IoU 0,9782.
- 3. `fullres_deeplabv3plus_resnet18_focal_dice_rgb_seed42` (candidato): DeepLabV3+ / resnet18, features `rgb`, loss `focal_dice`, val macro IoU 0,9978, test macro IoU 0,9992.
- 4. `fullres_unetplusplus_resnet18_focal_lovasz_rgb_indices_seed42` (candidato): U-Net++ / resnet18, features `rgb_indices`, loss `focal_lovasz`, val macro IoU 0,9973, test macro IoU 0,9767.
- 5. `fullres_unet_mit_b0_focal_lovasz_rgb_seed42` (candidato): U-Net / mit_b0, features `rgb`, loss `focal_lovasz`, val macro IoU 0,9908, test macro IoU 0,9920.

## Entradas e classes

- Polígonos de treinamento: `IBGE_method/own_LULC/extra_data/Classes_Uso_Solo.gpkg`.
- Ortofoto RGB/RGBA: `/home/kaue/data/landslide/feb26/Ortho_4_GNSS-AAT_16cm.tif`.
- Campo textual: `Classe`.
- Campo numérico: `Class_num`.
- Pixels ignorados no treinamento: `255`.
- Nodata no raster LULC final: `0`.

O treinamento usa classes internas 0 a 4, mas o raster final usa os códigos originais 1 a 5. O valor 255 é reservado para pixels ignorados durante o treinamento; o valor 0 é nodata de saída. Essa separação evita confundir ausência de dado com uma classe semântica real.

- Classe 1: `artif`, nota IBGE USOVEG 10,0.
- Classe 2: `descob`, nota IBGE USOVEG 5,0.
- Classe 3: `corpo_agua`, nota IBGE USOVEG 0,0.
- Classe 4: `veg_campestre`, nota IBGE USOVEG 2,0.
- Classe 5: `veg_florestal`, nota IBGE USOVEG 1,0.

Para a álgebra IBGE, a conversão é direta: área artificial recebe nota 10, área descoberta nota 5, corpo d'água nota 0 e exclusão da máscara válida, vegetação campestre nota 2 e vegetação florestal nota 1.

## Leitura técnica do modelo selecionado

U-Net é a arquitetura de referência para segmentação supervisionada com poucos rótulos. O encoder `resnet34` fornece as feições profundas e o decoder reconstrói a máscara na resolução do tile. O encoder ResNet-34 aumenta a profundidade em relação ao ResNet-18. Ele foi usado para testar se mais capacidade convolucional melhora classes visualmente parecidas, como solo exposto e área artificial. O feature set RGB usa exclusivamente os três primeiros canais da ortofoto. Ele testa quanto o problema pode ser resolvido apenas pela aparência direta dos pixels. Cross-entropy ponderada preserva uma interpretação probabilística por pixel e compensa frequências de classe; Lovasz adiciona pressão direta sobre IoU. Essa combinação é um teste forte para classes raras.

O modelo selecionado combina U-Net com encoder ResNet-34, feature set RGB e perda `weighted_ce_lovasz`. Essa configuração foi a melhor por validação nos artefatos disponíveis. A escolha por validação preserva a função do teste como holdout final.

A U-Net reconstrói a máscara de segmentação por um decoder que recebe conexões de salto do encoder. Essas conexões preservam detalhes espaciais de bordas e objetos pequenos, relevantes em ortofoto de 16 cm. O ResNet-34 adiciona mais profundidade que o ResNet-18, aumentando capacidade para distinguir padrões visualmente próximos, como área artificial, solo descoberto e vegetação baixa.

A perda `weighted_ce_lovasz` combina cross-entropy ponderada por classe com Lovasz. A cross-entropy mantém estabilidade por pixel e compensa classes raras; Lovasz adiciona pressão direta sobre IoU. Essa combinação é coerente com o critério de seleção por macro IoU, pois o objetivo não é apenas acertar muitos pixels, mas equilibrar desempenho entre as cinco classes.

## Treinamento e avaliação

- Otimizador: `adamw`.
- Scheduler: `cosine`.
- Learning rate: `0.0003`.
- Weight decay: `0.0001`.
- Batch size: `8`.
- Épocas configuradas: `100`.
- Early stopping patience: `100`.
- Augmentation: flips 0.5, rotate90 `True`, brilho 0.08, contraste 0.12.

A divisão treino/validação/teste usa blocos espaciais estratificados por classe. Essa estratégia reduz vazamento espacial e evita que classes pequenas desapareçam dos splits de validação e teste. A macro IoU é a métrica principal porque dá peso equivalente às cinco classes, mesmo quando a área de cada uma é muito diferente.

## Inferência e uso no IBGE Lite

- Janela de inferência: `256` pixels.
- Sobreposição: `64` pixels.
- Batch de inferência: `8`.

O raster LULC individual selecionado é reamostrado para o grid do DTM por vizinho mais próximo, preservando classes categóricas. Em seguida, cada classe é convertida para a nota USOVEG e entra na álgebra IBGE com peso 0,10. Todos os demais temas permanecem iguais ao produto principal: DTM final, declividade derivada do drone, geologia, pedologia, geomorfologia e pluviosidade.

## O que não existe nesta versão lite

Esta versão não calcula agreement, confidence, margin ou entropy de ensemble. Esses diagnósticos dependem de múltiplos modelos e não fazem sentido para um único votante. A incerteza aqui deve ser inferida por inspeção visual, pelas métricas do holdout e pela comparação com o produto principal com ensemble.

## Observação sobre artefatos

Métricas do modelo individual lidas dos artefatos locais em outputs_fullres/.
