# Descrição completa do método LULC

Esta aba documenta o método de geração da camada LULC usada como tema USOVEG no produto IBGE adaptado. O objetivo do LULC foi substituir uma base regional ou nacional de uso e cobertura da terra por um produto local treinado com polígonos interpretados sobre ortofoto de drone.

A implementação está versionada no repositório, principalmente em `IBGE_method/own_LULC/lulc_inputs.py` e nos módulos de `IBGE_method/own_LULC/implementation/`. O arquivo `lulc_inputs.py` é a superfície única de hiperparâmetros: entradas, classes, resolução, divisão espacial, modelos, perdas, treino, inferência e ensemble ficam declarados ali. Isso é intencional: o relatório deve explicar uma configuração que pode ser reexecutada, e não uma sequência informal de decisões manuais.

Este método LULC tem uma função específica dentro do produto final. Ele não estima suscetibilidade diretamente. Ele classifica cobertura e uso da terra em cinco classes locais; depois, essas classes são convertidas para a nota USOVEG da álgebra IBGE. Portanto, o LULC influencia o score final apenas pelo peso de USOVEG, mas influencia também a máscara válida porque corpos d'água são excluídos da suscetibilidade.

## Leitura recomendada da aba

A primeira parte descreve o dado de entrada, a codificação das classes e o desenho de avaliação. A segunda parte descreve as cinco redes individuais, porque cada uma foi incluída e como ler suas métricas. A terceira parte descreve o ensemble, que é o produto de produção usado pelo IBGE adaptado. As tabelas no final reproduzem os números dos arquivos JSON disponíveis localmente; se os artefatos da máquina externa forem substituídos depois, o site passa a refletir os novos números na próxima geração.

## Entradas e codificação de classes

- Polígonos de treinamento: `IBGE_method/own_LULC/extra_data/Classes_Uso_Solo.gpkg`.
- Ortofoto RGB/RGBA: `/home/kaue/data/landslide/feb26/Ortho_4_GNSS-AAT_16cm.tif`.
- Campo textual: `Classe`.
- Campo numérico: `Class_num`.
- Pixels ignorados no treinamento: `255`.
- Nodata no raster LULC final: `0`.

As classes usadas no raster final são os códigos originais 1 a 5. Internamente, durante o treinamento, elas são convertidas para índices 0 a 4 porque a função de perda multiclasse espera classes contíguas. Na escrita do GeoTIFF final, a codificação volta para 1 a 5. Esse detalhe é importante para auditoria: as métricas internas por classe normalmente aparecem como 0, 1, 2, 3 e 4, enquanto o GeoTIFF final e a álgebra IBGE usam 1, 2, 3, 4 e 5.

O valor 255 é usado apenas para pixels ignorados no treinamento. Isso inclui pixels fora dos polígonos rotulados, pixels inválidos da ortofoto e regiões onde não se quer calcular perda. Esse valor nunca deve aparecer como classe LULC final. O valor 0 é reservado para nodata de saída, isto é, pixels onde o modelo não deve produzir uma classe válida.

- Classe 1: `artif`, nota IBGE USOVEG 10,0.
- Classe 2: `descob`, nota IBGE USOVEG 5,0.
- Classe 3: `corpo_agua`, nota IBGE USOVEG 0,0.
- Classe 4: `veg_campestre`, nota IBGE USOVEG 2,0.
- Classe 5: `veg_florestal`, nota IBGE USOVEG 1,0.

A interpretação semântica das classes foi mantida curta no raster para evitar ambiguidades: `artif` representa áreas artificiais, `descob` representa solo descoberto ou superfícies expostas não vegetadas, `corpo_agua` representa água, `veg_campestre` representa vegetação baixa/campestre e `veg_florestal` representa vegetação arbórea/florestal. Para o IBGE adaptado, essas classes não entram como nomes; entram como notas USOVEG: artificial recebe nota alta, solo descoberto recebe nota intermediária, vegetação recebe notas baixas e água é excluída.

## Conjuntos de variáveis

Foram previstos dois conjuntos de atributos derivados apenas da ortofoto, sem insumos externos adicionais:

- `rgb`: red, green, blue.
- `rgb_indices`: red, green, blue, hue, saturation, value, brightness, excess_green, local_texture.

O conjunto `rgb` preserva o problema na forma mais direta possível: o modelo vê vermelho, verde e azul e precisa aprender as separações a partir da aparência. Ele é simples, barato e reduz o risco de introduzir transformações artificiais que distorçam os dados.

O conjunto `rgb_indices` amplia o RGB com HSV, brilho, excesso de verde e textura local. Ele foi incluído para ajudar a separar vegetação, solo exposto, água e superfícies artificiais quando a assinatura espectral RGB pura é ambígua. O excesso de verde favorece a separação de vegetação; HSV explicita matiz e saturação; brilho ajuda em sombras e superfícies claras; textura local ajuda a diferenciar telhados, copas, solo exposto e áreas homogêneas de água.

A quarta banda da ortofoto, quando existe, não é usada como classe ou feição espectral principal. Ela pode atuar como máscara de validade, evitando que regiões sem imagem útil entrem no treinamento ou na inferência. A decisão de usar apenas os três primeiros canais como RGB reduz dependência de um canal alfa que pode variar entre exportações de ortofoto.

## Divisão espacial e desenho de avaliação

- Estratégia de split: `stratified_spatial_blocks`.
- Percentuais por classe: treino 50,00%, validação 35,00%, teste 15,00%.
- Exigência de todas as classes em cada split: `True`.
- Tentativas máximas de seeds para achar split viável: `400`.

A divisão não é uma amostragem aleatória simples de pixels. Ela usa blocos espaciais para reduzir vazamento espacial entre treino, validação e teste. Se pixels vizinhos quase idênticos fossem sorteados aleatoriamente para splits diferentes, a validação poderia medir memorização local da ortofoto, e não generalização para blocos realmente não vistos.

Além disso, as metas de proporção são aplicadas por classe. Isso significa que a distribuição desejada de treino, validação e teste é verificada dentro de cada classe, não apenas no total de pixels. Essa escolha foi feita depois de observar que classes minoritárias podiam ficar com suporte muito baixo em validação ou teste. Sem suporte mínimo por classe, a macro IoU fica instável: uma classe pequena pode dominar a interpretação da métrica por acaso, ou simplesmente não ser avaliada de maneira útil.

A validação é usada para seleção de modelos e escolha do ensemble. O teste é mantido como confirmação final. Essa separação é deliberada: olhar o teste para escolher modelo causaria viés de seleção. No relatório, por isso, a macro IoU de validação explica por que um modelo foi considerado forte, enquanto a macro IoU de teste indica se essa escolha se confirmou em blocos mantidos fora da seleção.

A métrica principal é macro IoU em cinco classes. Macro IoU calcula IoU por classe e depois tira a média simples, dando o mesmo peso a classes grandes e pequenas. Isso é mais exigente do que acurácia global, porque um modelo não pode compensar desempenho ruim em `descob` ou `corpo_agua` acertando muitos pixels de vegetação.

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

A ponderação por classe e o sampler balanceado atacam problemas diferentes. A ponderação na função de perda aumenta o custo de errar classes raras nos pixels rotulados. O sampler aumenta a chance de essas classes raras aparecerem nos batches. Em conjunto, eles evitam que o treinamento seja dominado pela classe espacialmente mais abundante.

Os aumentos de dados foram mantidos conservadores. Flips e rotações de 90 graus são adequados porque a classe de uso e cobertura não depende da orientação absoluta da imagem. Pequenas mudanças de brilho e contraste simulam diferenças de iluminação e exposição. Borramento e ruído existem como hiperparâmetros, mas ficam desabilitados por padrão para não degradar bordas finas em uma ortofoto de 16 cm.

O otimizador AdamW foi usado porque separa atualização de pesos e decaimento, sendo estável em redes de segmentação modernas. O scheduler cosseno reduz a taxa de aprendizado progressivamente, permitindo passos maiores no início e ajustes mais finos no final das 100 épocas.

## Funções de perda testadas

Foram usadas combinações de perdas complementares. Cross-entropy ponderada é uma perda por pixel estável e probabilística; focal loss reduz o peso de exemplos fáceis e enfatiza erros difíceis; dice loss favorece sobreposição espacial; Lovasz aproxima diretamente uma otimização ligada à IoU. Como o objetivo de seleção é macro IoU, Lovasz é especialmente útil nas configurações em que a prioridade é melhorar interseção sobre união, e não apenas acurácia por pixel.

A combinação `weighted_ce_dice` foi usada como linha de base. `focal_dice` foi testada para tornar o treino mais sensível a exemplos difíceis sem abandonar a ideia de sobreposição. `weighted_ce_lovasz` e `focal_lovasz` foram incluídas para pressionar diretamente a métrica de IoU, o que faz sentido quando classes pequenas precisam aparecer bem na avaliação macro.

## Inferência full-resolution

- Janela de inferência: `256` pixels.
- Sobreposição: `64` pixels.
- Batch de inferência: `8`.
- CUDA habilitado quando disponível: `True`.

A inferência percorre a ortofoto em janelas sobrepostas. Para cada pixel, as probabilidades acumuladas nas janelas que o cobrem são médias antes da decisão final. A sobreposição reduz artefatos de borda, porque pixels próximos às margens de uma janela também são vistos em outra janela, mais centralizada.

O resultado individual de cada modelo inclui um raster de classes e um raster multibanda de probabilidades, com cinco bandas, uma por classe. O raster de classes é útil para inspeção direta. O raster de probabilidades é mais importante para o ensemble, porque preserva incerteza: um modelo que prevê classe 4 com probabilidade 0,51 carrega informação diferente de um modelo que prevê classe 4 com probabilidade 0,99.

## Por que cinco modelos diferentes

O ensemble não foi desenhado para juntar cinco cópias quase idênticas do mesmo treinamento. Ele combina arquiteturas, encoders, losses e feature sets diferentes. A motivação é reduzir dependência de um único viés de modelo. Quando modelos diferentes convergem para a mesma classe em um pixel, a confiança interpretativa aumenta. Quando divergem, os rasters de concordância, margem e entropia indicam onde o mapa merece inspeção humana.

A diversidade arquitetural cobre quatro famílias: U-Net++, DeepLabV3+, U-Net e FPN. A diversidade de encoder cobre ResNet-18, ResNet-34 e MiT-B0. A diversidade de perda cobre focal, dice, Lovasz e cross-entropy ponderada. A diversidade de features aparece principalmente no modelo U-Net++ com `rgb_indices`, enquanto os demais mantêm RGB para testar robustez sem engenharia adicional de canais.

## Os cinco modelos full-res

- Modelo 1: U-Net++ / resnet18.
- Identificador: `fullres_unetplusplus_resnet18_focal_lovasz_rgb_indices_seed42`.
- Configuração: resolução 0,16 m; feature set `rgb_indices`; perda `focal_lovasz`; tile 192; stride 96; seed 42.
- Treinamento: 100 épocas planejadas/executadas; melhor época 61.
- Validação: macro IoU 0,9973, macro F1 0,9987, acurácia global 0,9996.
- Teste final: macro IoU 0,9767, macro F1 0,9880, acurácia global 0,9980.
- Leitura técnica: U-Net++ usa conexões de salto aninhadas entre encoder e decoder. A intenção é reduzir a lacuna semântica entre detalhes rasos da imagem e representações profundas, o que tende a ajudar em bordas finas entre telhados, solo exposto, vegetação e pequenos corpos d'água. O encoder ResNet-18 foi escolhido como opção leve e robusta, com boa relação entre capacidade e custo computacional. Em full-res, essa escolha é importante porque cada época processa muitos tiles. O feature set RGB+índices adiciona HSV, brilho, excesso de verde e textura local. Isso torna explícitas pistas que o modelo poderia aprender sozinho, mas que ajudam quando há poucos polígonos rotulados. A perda focal reduz a influência de pixels fáceis e coloca mais peso nos erros difíceis; Lovasz aproxima diretamente uma otimização orientada a IoU. A combinação é adequada quando a métrica de seleção é macro IoU.

- Modelo 2: DeepLabV3+ / resnet18.
- Identificador: `fullres_deeplabv3plus_resnet18_focal_dice_rgb_seed42`.
- Configuração: resolução 0,16 m; feature set `rgb`; perda `focal_dice`; tile 192; stride 96; seed 42.
- Treinamento: 100 épocas planejadas/executadas; melhor época 68.
- Validação: macro IoU 0,9978, macro F1 0,9989, acurácia global 0,9997.
- Teste final: macro IoU 0,9992, macro F1 0,9996, acurácia global 0,9996.
- Leitura técnica: DeepLabV3+ combina encoder convolucional com pirâmide espacial atrous. Ele foi incluído porque observa o contexto em múltiplas escalas sem perder completamente a resolução, uma propriedade útil quando a mesma classe aparece em manchas pequenas e grandes. O encoder ResNet-18 foi escolhido como opção leve e robusta, com boa relação entre capacidade e custo computacional. Em full-res, essa escolha é importante porque cada época processa muitos tiles. O feature set RGB usa exclusivamente os três primeiros canais da ortofoto. Ele testa quanto o problema pode ser resolvido apenas pela aparência direta dos pixels. Focal+dice combina foco em exemplos difíceis com sobreposição espacial. Ela é útil para classes pequenas porque a componente dice não deixa a otimização ser dominada apenas pela frequência dos pixels.

- Modelo 3: U-Net / resnet34.
- Identificador: `fullres_unet_resnet34_weighted_ce_lovasz_rgb_seed42`.
- Configuração: resolução 0,16 m; feature set `rgb`; perda `weighted_ce_lovasz`; tile 192; stride 96; seed 42.
- Treinamento: 100 épocas planejadas/executadas; melhor época 100.
- Validação: macro IoU 0,9992, macro F1 0,9996, acurácia global 0,9997.
- Teste final: macro IoU 0,9956, macro F1 0,9978, acurácia global 0,9974.
- Leitura técnica: U-Net é a arquitetura de referência para segmentação supervisionada com poucos rótulos. O encoder `resnet34` fornece as feições profundas e o decoder reconstrói a máscara na resolução do tile. O encoder ResNet-34 aumenta a profundidade em relação ao ResNet-18. Ele foi usado para testar se mais capacidade convolucional melhora classes visualmente parecidas, como solo exposto e área artificial. O feature set RGB usa exclusivamente os três primeiros canais da ortofoto. Ele testa quanto o problema pode ser resolvido apenas pela aparência direta dos pixels. Cross-entropy ponderada preserva uma interpretação probabilística por pixel e compensa frequências de classe; Lovasz adiciona pressão direta sobre IoU. Essa combinação é um teste forte para classes raras.

- Modelo 4: U-Net / mit_b0.
- Identificador: `fullres_unet_mit_b0_focal_lovasz_rgb_seed42`.
- Configuração: resolução 0,16 m; feature set `rgb`; perda `focal_lovasz`; tile 192; stride 96; seed 42.
- Treinamento: 100 épocas planejadas/executadas; melhor época 99.
- Validação: macro IoU 0,9908, macro F1 0,9953, acurácia global 0,9970.
- Teste final: macro IoU 0,9920, macro F1 0,9960, acurácia global 0,9990.
- Leitura técnica: U-Net é a arquitetura de referência para segmentação supervisionada com poucos rótulos. O encoder `mit_b0` fornece as feições profundas e o decoder reconstrói a máscara na resolução do tile. O encoder MiT-B0 introduz uma alternativa leve baseada em atenção/transformer. Ele não é apenas mais um ResNet: sua função no conjunto é trazer um viés arquitetural diferente, potencialmente mais sensível a relações espaciais amplas. O feature set RGB usa exclusivamente os três primeiros canais da ortofoto. Ele testa quanto o problema pode ser resolvido apenas pela aparência direta dos pixels. A perda focal reduz a influência de pixels fáceis e coloca mais peso nos erros difíceis; Lovasz aproxima diretamente uma otimização orientada a IoU. A combinação é adequada quando a métrica de seleção é macro IoU.

- Modelo 5: FPN / resnet34.
- Identificador: `fullres_fpn_resnet34_weighted_ce_lovasz_rgb_seed42`.
- Configuração: resolução 0,16 m; feature set `rgb`; perda `weighted_ce_lovasz`; tile 192; stride 96; seed 42.
- Treinamento: 100 épocas planejadas/executadas; melhor época 77.
- Validação: macro IoU 0,9983, macro F1 0,9991, acurácia global 0,9990.
- Teste final: macro IoU 0,9782, macro F1 0,9888, acurácia global 0,9972.
- Leitura técnica: FPN agrega mapas de feições em diferentes níveis de escala. Ele funciona como um contraponto aos decoders U-Net porque enfatiza uma pirâmide explícita de feições, favorecendo a estabilidade em objetos de tamanhos variados. O encoder ResNet-34 aumenta a profundidade em relação ao ResNet-18. Ele foi usado para testar se mais capacidade convolucional melhora classes visualmente parecidas, como solo exposto e área artificial. O feature set RGB usa exclusivamente os três primeiros canais da ortofoto. Ele testa quanto o problema pode ser resolvido apenas pela aparência direta dos pixels. Cross-entropy ponderada preserva uma interpretação probabilística por pixel e compensa frequências de classe; Lovasz adiciona pressão direta sobre IoU. Essa combinação é um teste forte para classes raras.

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

Na prática, a média de probabilidades responde a duas perguntas ao mesmo tempo. A primeira é qual classe recebeu maior suporte coletivo dos modelos. A segunda é quão concentrado foi esse suporte. Se todos os modelos favorecem a mesma classe com alta probabilidade, a confiança e a margem tendem a ser altas e a entropia tende a ser baixa. Se os modelos se dividem entre duas classes, a classe final ainda é definida, mas a margem diminui e a entropia aumenta.

A concordância por voto duro é uma métrica diferente da confiança probabilística. Ela mede a fração de modelos cuja classe mais provável coincide com a classe final do ensemble. Uma concordância baixa pode ocorrer mesmo quando a confiança média é alta, especialmente em pixels onde um modelo diverge sistematicamente dos demais ou onde probabilidades muito fortes de alguns modelos superam votos mais fracos de outros. Por isso, concordância, confiança, margem e entropia devem ser lidas em conjunto, não isoladamente.

A margem é a diferença entre a maior e a segunda maior probabilidade média. Ela é uma das métricas mais úteis para revisão visual: margens baixas indicam zonas de fronteira, mistura espectral ou ambiguidade semântica. A entropia normalizada mede dispersão geral entre as cinco classes. Entropia alta indica que o ensemble não concentrou probabilidade em uma única classe; nesses locais, a classificação final deve ser tratada como menos estável.

## Interpretação das métricas

A acurácia global mostra a fração de pixels avaliados corretamente, mas pode ser excessivamente otimista quando uma classe ocupa grande área. Macro F1 resume equilíbrio entre precisão e revocação por classe. Macro IoU é a métrica mais rígida e foi usada como principal critério porque penaliza simultaneamente falsos positivos e falsos negativos em cada classe.

Resultados muito altos em validação e teste indicam que os polígonos rotulados, o grid full-res e o desenho de split produziram um problema separável para os blocos avaliados. Isso é bom, mas não elimina a necessidade de auditoria visual. A avaliação mede desempenho contra os rótulos disponíveis; se os rótulos tiverem bordas imprecisas, classes semanticamente misturadas ou lacunas fora dos polígonos, essas limitações não aparecem automaticamente nas métricas.

No uso final, os rasters de diagnóstico do ensemble são tão importantes quanto o raster de classe. Áreas com baixa margem, baixa concordância ou entropia mais alta devem ser priorizadas em revisão humana, principalmente quando coincidem com transições entre vegetação campestre, vegetação florestal, solo exposto e áreas artificiais.

## Relação com o IBGE adaptado

O LULC entra no IBGE adaptado como USOVEG. A conversão é feita depois da classificação: classe 1 vira nota 10, classe 2 vira nota 5, classe 3 vira nota 0 e é excluída da máscara válida, classe 4 vira nota 2 e classe 5 vira nota 1. Essa transformação é deliberadamente simples para manter rastreabilidade entre o mapa de cobertura e a álgebra IBGE.

Como o peso de USOVEG na álgebra final é 0,10, erros de LULC não dominam sozinhos o score IBGE. Ainda assim, podem alterar o resultado em zonas urbanizadas, solo exposto e bordas de água. A classe de água é especialmente sensível porque afeta a máscara válida, não apenas a nota ponderada.

## Limitações e cuidados

O treinamento depende dos polígonos disponíveis. Áreas não representadas nos polígonos podem ser classificadas com menor confiabilidade, mesmo que a métrica em validação e teste seja alta. A ortofoto também representa uma data específica; mudanças posteriores de uso do solo não são capturadas sem nova imagem ou nova inferência.

O produto full-res tem granularidade de 16 cm, mas a semântica das classes não deve ser interpretada como verdade absoluta em cada pixel isolado. Em ortofotos de alta resolução, bordas de copa, sombras, transições solo-vegetação e superfícies parcialmente cobertas podem misturar respostas espectrais. Por isso, a leitura mais robusta é espacial: padrões contínuos e zonas de incerteza são mais informativos do que pixels isolados.

## Reprodutibilidade

A execução full-res foi preparada para máquina externa com CUDA. O script `IBGE_method/own_LULC/run_fullres_external.sh` recebe a ortofoto como argumento, exporta os caminhos necessários e executa a varredura full-res com retomada segura. O ensemble pode ser reconstruído a partir dos rasters de probabilidade já gerados, sem retreinar os modelos, desde que os artefatos de cada votante estejam presentes.

Os principais arquivos de controle são `lulc_inputs.py`, `sweep_results.json`, `ensemble_results.json` e `selected_experiment.json`. O primeiro descreve o que deveria ser executado; os três últimos descrevem o que foi efetivamente executado e selecionado. Esta aba lê esses artefatos e evita copiar modelos ou GeoTIFFs para o website.

## Observação sobre placeholders

Métricas lidas dos artefatos locais em outputs_fullres/.
