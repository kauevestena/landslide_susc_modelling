# Mapeamentos proxy de Geologia, Geomorfologia e Pedologia

Este documento audita os mapeamentos proxy usados nos temas GEO, GEM e PED do produto IBGE adaptado em 16 cm. O objetivo é explicar o que foi efetivamente implementado, quais critérios metodológicos vieram da revisão do método IBGE, quais inferências foram feitas para adaptar os arquivos locais CPRM/SGB/SIG, e quais limitações permanecem.

A leitura mais importante é esta: os três temas preservam a lógica direcional da metodologia IBGE, mas não afirmam reproduzir integralmente a base nacional BDIA do IBGE. A adaptação usa os dados locais disponíveis, rasteriza as notas diretamente no grid do DTM final de 16 cm, e registra os valores de entrada no provenance. Quando uma classe não mapeada aparece dentro do recorte do DTM, o pipeline falha em vez de preencher silenciosamente.

## Escopo e transparência

- Este texto documenta a versão adaptada de alta resolução, não o produto nacional estrito em grade estatística de 1 km.
- O grid de referência é o DTM final do drone em 16 cm.
- Geologia, Geomorfologia e Pedologia entram como proxies locais compatíveis, porque os arquivos disponíveis não expõem exatamente todas as tabelas e campos intermediários da metodologia BDIA nacional.
- As notas são tratadas como graus de potencialidade 1-10. Valores maiores indicam maior contribuição temática à suscetibilidade.
- A documentação separa fato implementado, justificativa técnica e limitação. Isso evita transformar uma decisão proxy em uma afirmação de equivalência perfeita com o IBGE estrito.

## Como o IBGE estrito faria

No método nacional revisado em `IBGE_method/IBGE_method_full_review.md`, os três temas não são simples nomes de classes. Eles são interpretações temáticas padronizadas em graus de potencialidade.

- **GEO / Geologia:** o IBGE calcula a nota geológica a partir da média de quatro componentes: litologia, característica genética da litologia, província estrutural e subprovíncia estrutural. A mesma rocha pode ter comportamento distinto dependendo da gênese, estrutura e deformação tectônica.
- **GEM / Geomorfologia:** o IBGE usa modelados geomorfológicos, especialmente a quarta ordem taxonômica da BDIA. Modelados de acumulação tendem a notas menores; modelados de dissecação, com maior densidade de drenagem e aprofundamento das incisões, tendem a notas maiores.
- **PED / Pedologia:** o IBGE usa o solo dominante da unidade de mapeamento e calcula `PED = max(PROF, TEXT, RELTEXT)`. O máximo representa a característica pedológica mais restritiva entre profundidade, textura e relação ou gradiente textural.

## Como a adaptação local fez

- A base nacional estrita não foi reproduzida campo a campo. Em vez disso, foram usados shapefiles locais CPRM/SGB/SIG com campos operacionais compatíveis.
- As camadas categóricas foram rasterizadas diretamente no grid do DTM de 16 cm e alinhadas por vizinho mais próximo.
- O código recorta cada shapefile pelo bbox do DTM antes de validar os valores. Portanto, classes presentes no shapefile municipal, mas fora do recorte final, não entram no produto final nem na tabela `mapped_values` dos relatórios.
- Se algum valor aparece dentro do bbox do DTM e não existe no dicionário de notas, a geração falha com erro explícito. Não há fallback silencioso para zero, média ou classe vizinha.
- O dicionário versionado contém mais classes do que as que aparecem no recorte final. Isso é intencional: permite reutilização em áreas adjacentes sem mudar o código, mas o relatório do produto mostra apenas as classes efetivamente usadas no grid final.

## Valores existentes nos shapefiles e valores efetivamente usados

- Geologia: shapefile `/home/kaue/Downloads/05_Geospatial_Data_Maps/sig_cachoeirodoitapemirim_es_suscet/data_interest_area/06.Geologia/Geologia.shp`, campo `SIGLA_UNID`. Valores existentes no shapefile: `C_cortado_2a_gamma_5Isa`, `NP3a_gamma_1Iag`, `NP3a_gamma_1Ich`, `NP3a_gamma_1Imf`, `PRps`, `Q2a`, `Q2fl`. Valores que interceptaram o grid final: `PRps`.
- Geomorfologia: shapefile `/home/kaue/Downloads/05_Geospatial_Data_Maps/sig_cachoeirodoitapemirim_es_suscet/data_interest_area/04.PadrõesDeRelevo/Relevo.shp`, campo `Classe`. Valores existentes no shapefile: `Colinas`, `Morros altos`, `Morros baixos`, `Morrotes`, `Morrotes altos`, `Planícies e terraços fluviais`, `Serras`. Valores que interceptaram o grid final: `Colinas`, `Morros altos`, `Morros baixos`, `Planícies e terraços fluviais`.
- Pedologia: shapefile `/home/kaue/Downloads/05_Geospatial_Data_Maps/sig_cachoeirodoitapemirim_es_suscet/data_interest_area/07.Pedologia/Pedologia.shp`, campo `DESC_`. Valores existentes no shapefile: `Ad2`, `BV1`, `BV2`, `Ca3`, `LVa2`, `PEe1`, `PEe11`, `PEe3`, `PEe4`, `Ra`, `Rde1`, `Rios`. Valores que interceptaram o grid final: `PEe1`, `Rios`.

## Tabela implementada: Geologia / GEO

Dicionário versionado completo em `src/three_method_comparison.py`:

- `C_cortado_2a_gamma_5Isa` -> 6,00.
- `NP3a_gamma_1Iag` -> 6,75.
- `NP3a_gamma_1Ich` -> 6,75.
- `NP3a_gamma_1Imf` -> 6,75.
- `PRps` -> 7,30.
- `Q2a` -> 9,50.
- `Q2fl` -> 9,30.

Valores efetivamente usados nesta execução:

- `PRps` -> 7,30.

A camada GEO gerada teve 44.972.904 pixels avaliados, variou de 7,3000 a 7,3000, média 7,3000 e desvio padrão 0,0000. No recorte final, a nota ficou espacialmente constante porque apenas `PRps` interceptou o grid válido.

### Critério usado para `PRps -> 7,3`

`SIGLA_UNID` foi usado porque é o identificador sintético da unidade geológica. Ele preserva a unidade de mapeamento como objeto interpretável, enquanto campos como litotipo, deformação, fraturamento, coerência, relevo e declividade descrevem propriedades internas dessa unidade. A nota `7,3` representa uma condição alta/intermediária-alta, não extrema.

A decisão é coerente com os atributos locais observados para `PRps`: a unidade Paraíba do Sul aparece como complexo metamórfico, com gnaisse milonítico, granada gnaisse, metamarga e litotipos associados; apresenta deformação dúctil/rúptil, fraturamento moderado a intenso, comportamento anisotrópico, baixa a alta alteração/intemperismo e associação frequente a domínio montanhoso ou morros/serras. Esses elementos são compatíveis com maior fragilidade estrutural e maior possibilidade de planos de fraqueza quando comparados a unidades maciças pouco deformadas.

Ao mesmo tempo, a nota não foi levada para 9 ou 10 porque GEO, no método IBGE, não deve duplicar integralmente o efeito da declividade e da geomorfologia. O relevo íngreme e a inclinação entram de forma própria em GEM e DECL. A nota geológica deve representar predisposição litológico-estrutural, não o score final de encosta.

## Atributos CPRM/SGB observados para as classes geológicas usadas

- Classe `PRps`:
- NOME_UNIDA: `Paraíba do Sul`.
- HIERARQUIA: `Complexo`.
- LITOTIPO1: `Gnaisse milonítico, Granada gnaisse e Metamarga`.
- LITOTIPO2: `Anfibolito, Mármore e Quartzito`.
- CLASSE_ROC: `Metamórfica`.
- DEF_TEC: `Intensamente dobrada`.
- CIS_FRAT: `Moderadamente a intensamente fraturada (Distribuição irregular)`.
- TIPO_DEF: `Deformação dúctil/rúptil`.
- COMP_REOL: `Anisotrópico`.
- RELEVO: `Domínio Montanhoso`, `Domínio de Colinas Dissecadas e de Morros Baixos`, `Domínio de Morros e de Serras Baixas`.
- DECLIVIDAD: `15 a 35`, `25 a 45`, `5 a 20`.
- AMPL_TOPO: `30 a 80 metros`, `80 a 200 metros`, `acima de 300 metr*`.

## Limitações específicas da Geologia

- O recorte final usa apenas `PRps`. Isso reduz a capacidade da geologia de diferenciar subáreas dentro do produto.
- A adaptação não recalcula explicitamente `GEO = média(LIT, GEN, PROV, SUBPROV)` porque a tabela BDIA completa com esses componentes normalizados não está disponível neste fluxo local.
- O mapeamento é rastreável e coerente com atributos CPRM/SGB, mas deveria ser revisado por geólogo se o produto for usado como laudo formal.

## Tabela implementada: Geomorfologia / GEM

Dicionário versionado completo em `src/three_method_comparison.py`:

- `Colinas` -> 3,00.
- `Morros altos` -> 9,00.
- `Morros baixos` -> 7,00.
- `Morrotes` -> 5,00.
- `Morrotes altos` -> 6,00.
- `Planícies e terraços fluviais` -> 1,00.
- `Serras` -> 10,00.

Valores efetivamente usados nesta execução:

- `Colinas` -> 3,00.
- `Morros altos` -> 9,00.
- `Morros baixos` -> 7,00.
- `Planícies e terraços fluviais` -> 1,00.

A camada GEM gerada teve 44.972.904 pixels avaliados, variou de 1,0000 a 9,0000, média 6,6379 e desvio padrão 3,4520. Diferentemente de GEO e PED, GEM apresentou variação espacial relevante dentro do recorte.

### Critério geomorfológico das notas

`Classe` foi usado porque o shapefile de padrões de relevo já entrega uma classificação morfológica direta: planícies, colinas, morros baixos, morros altos, morrotes e serras. Essa legenda não é a tabela BDIA de modelados, mas é o proxy local mais próximo para expressar energia do relevo, dissecação e posição morfológica.

- `Planícies e terraços fluviais -> 1`: representa ambiente baixo, deposicional ou de acumulação, com baixa energia de relevo. Na lógica IBGE, modelados de acumulação tendem a notas menores.
- `Colinas -> 3`: representa relevo ondulado ou dissecado de menor energia. A nota baixa a moderada reconhece que há forma de relevo, mas sem atribuir o mesmo potencial de morros ou serras.
- `Morros baixos -> 7`: representa relevo mais dissecado, com maior amplitude e maior energia que colinas. A nota alta reflete maior potencial geomorfológico para concentração de fluxos e instabilidade de encostas.
- `Morros altos -> 9`: representa relevo de alta energia, maior amplitude e formas mais favoráveis a encostas longas/íngremes. A nota muito alta é coerente com a direção IBGE para modelados de dissecação mais intensa.

A progressão 1, 3, 7 e 9 foi escolhida para manter um gradiente monotônico claro: quanto maior a energia do relevo e a dissecação, maior a nota GEM. Essa camada é uma das mais importantes para a adaptação local porque traz variação espacial regional que complementa a declividade de 16 cm calculada diretamente do DTM.

## Limitações específicas da Geomorfologia

- A legenda `Classe` não contém explicitamente os códigos de modelados BDIA usados no IBGE estrito.
- A nota não é calculada por densidade de drenagem e aprofundamento de incisões numéricos; ela é uma equivalência ordinal entre padrões de relevo locais e o sentido físico da tabela IBGE.
- Apesar disso, GEM é o proxy mais estável entre os três temas aqui auditados, porque a classe de relevo tem correspondência direta com energia geomorfológica e apresenta variação dentro do recorte.

## Tabela implementada: Pedologia / PED

Dicionário versionado completo em `src/three_method_comparison.py`:

- `Ad2` -> 5,00.
- `BV1` -> 7,00.
- `BV2` -> 7,00.
- `Ca3` -> 9,00.
- `LVa2` -> 7,00.
- `PEe1` -> 8,00.
- `PEe11` -> 8,00.
- `PEe3` -> 8,00.
- `PEe4` -> 8,00.
- `Ra` -> 10,00.
- `Rde1` -> 10,00.
- `Rios` -> 0,00.

Valores efetivamente usados nesta execução:

- `PEe1` -> 8,00.
- `Rios` -> 0,00.

A camada PED gerada teve 44.420.881 pixels avaliados, variou de 8,0000 a 8,0000, média 8,0000 e desvio padrão 0,0000. No recorte válido, a classe pedológica dominante foi `PEe1`; `Rios` recebeu nota 0 e entrou como exclusão/sem avaliação pedológica.

### Critério usado para `PEe1 -> 8`

`DESC_` foi usado como legenda pedológica operacional porque o shapefile local não expõe, de forma equivalente ao fluxo BDIA completo, os campos já normalizados para profundidade, textura e relação textural. O campo preserva a unidade de solo mapeada e permite uma nota proxy auditável.

`PEe1` foi tratado como classe restritiva e recebeu nota 8. A justificativa é coerente com a lógica pedológica do IBGE: Argissolos e classes com forte contraste textural ou gradiente entre horizontes tendem a aumentar restrições à infiltração/percolação e podem favorecer planos de descontinuidade hidrológica. Como o IBGE usa o máximo entre profundidade, textura e relação textural, uma característica restritiva basta para elevar a nota PED.

`Rios -> 0` não representa solo estável. Representa ausência de avaliação pedológica para água/canal. No produto final, água e classes não avaliáveis são tratadas como exclusão da máscara válida quando apropriado, não como terreno de baixa suscetibilidade.

## Atributos CPRM/SGB observados para as classes pedológicas usadas

- Classe `PEe1`:
- APTPCT: `P-100%`.
- P: `100`.
- SBCS: `PEe1`.
- EDAFICA: `P`.
- Fonte: `Manzatto et al. (2009)`.
- Escala: `1:250.000`.

- Classe `Rios`:
- SBCS: `Rios`.
- EDAFICA: `AGUA`.
- Fonte: `Manzatto et al. (2009)`.
- Escala: `1:250.000`.

## Limitações específicas da Pedologia

- O produto não recalcula `PED = max(PROF, TEXT, RELTEXT)` a partir dos três atributos originais do IBGE, porque esses campos não estão disponíveis no shapefile local no mesmo formato operacional.
- A presença dominante de `PEe1` torna a camada pouco variável dentro do recorte; isso é uma limitação temática do dado disponível, não do grid de 16 cm.
- A nota 8 é tecnicamente defensável como proxy restritivo, mas a calibração ideal exigiria tabela pedológica oficial completa ou revisão pericial por pedólogo.

## Regra de validação aplicada pelo pipeline

A validação operacional ocorre antes da rasterização em tiles. O código recorta cada shapefile pelo bbox do DTM, coleta os valores do campo de interesse e compara esses valores com o dicionário versionado. Se existir valor não mapeado dentro do bbox, a execução falha. Isso protege o produto contra uma classe nova entrar como zero, nodata ou média por acidente.

Depois da validação, as geometrias são rasterizadas por tile, usando a transformada e resolução do DTM. Como GEO, GEM e PED são variáveis categóricas já convertidas para nota, não há interpolação bilinear nesses temas. O valor final de cada pixel é a nota do polígono que cobre aquele pixel.

## Síntese de confiança

- **Geomorfologia:** confiança relativa mais alta entre os três proxies, porque a legenda local tem relação direta com energia do relevo e apresenta variação espacial no recorte.
- **Geologia:** confiança moderada. A interpretação de `PRps` é coerente com atributos geológico-estruturais, mas a camada é constante no recorte final e não substitui a média BDIA formal dos quatro componentes geológicos.
- **Pedologia:** confiança moderada a cautelosa. A nota de `PEe1` segue a lógica restritiva do IBGE, mas os campos completos `PROF`, `TEXT` e `RELTEXT` não foram recalculados formalmente.

## Recomendações futuras

- Obter as tabelas BDIA oficiais completas para geologia e pedologia, com os campos necessários para reproduzir `GEO` e `PED` de forma estrita.
- Anexar uma tabela pericial revisada por geólogo e pedólogo, mantendo as notas usadas e a justificativa de cada classe.
- Expandir o relatório com mapas de cobertura dos polígonos originais e estatísticas por classe quando o recorte de estudo for ampliado.
- Manter a regra atual de falhar em classes não mapeadas, pois ela é essencial para rastreabilidade.
