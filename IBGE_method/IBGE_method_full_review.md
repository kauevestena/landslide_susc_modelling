## 1. Ideia central da metodologia

O IBGE trabalha com seis temas:

1. **Geologia**
2. **Geomorfologia**
3. **Pedologia**
4. **Cobertura e uso da terra + Vegetação**
5. **Declividade**
6. **Pluviosidade**

Cada tema recebe um **grau de potencialidade a deslizamentos**, adimensional, em geral de **1 a 10**, em que valores maiores significam maior propensão. Depois os temas são integrados na **Grade Estatística do IBGE**, com células de **1 km x 1 km**, usando o critério de **máxima sobreposição**, isto é, o atributo que ocupa a maior área dentro da célula passa a representar aquela célula. Em seguida, os dados já agregados à grade são convertidos para raster e processados por álgebra de mapas. 

A equação final é uma **média ponderada**:

[
S = 0{,}15GEO + 0{,}20GEM + 0{,}15PED + 0{,}10USOVEG + 0{,}35DECL + 0{,}05PLUV
]

ou, equivalentemente:

[
S = \frac{15GEO + 20GEM + 15PED + 10USOVEG + 35DECL + 5PLUV}{100}
]

Os pesos finais escolhidos foram: **Geologia = 15**, **Geomorfologia = 20**, **Pedologia = 15**, **Cobertura/uso + Vegetação = 10**, **Declividade = 35** e **Pluviosidade = 5**. 

As classes finais são:

| Valor final (S) | Classe      |
| --------------: | ----------- |
|     0,00 – 3,50 | Muito baixa |
|     3,51 – 4,50 | Baixa       |
|     4,51 – 5,50 | Média       |
|     5,51 – 6,50 | Alta        |
|    6,51 – 10,00 | Muito alta  |

Essas classes foram definidas por **Quebras Naturais/Jenks** após simulações com diferentes pesos, avaliadas pela equipe técnica. O documento informa que o menor valor encontrado foi 0,70 e o maior 9,25. 

---

## 2. Camadas de entrada necessárias

### 2.1 Geologia

**Entrada:** base vetorial de Geologia do IBGE/BDIA, escala 1:250.000.

A geologia não é tratada apenas como “tipo de rocha”. O IBGE considera que a mesma litologia pode ter comportamentos diferentes dependendo da gênese, da estrutura e da deformação tectônica. Por isso, o tema Geologia usa quatro variáveis:

| Variável geológica                   | Como vira nota                                                                                                            |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| Litologia                            | Cada rocha recebe grau 1–10; se a unidade geológica tem várias litologias, calcula-se a média das litologias componentes. |
| Característica genética da litologia | Ex.: plutônica, sedimentar clástica, metamórfica regional etc.; cada classe recebe grau 1–10.                             |
| Província estrutural                 | Grau associado à província tectonoestrutural.                                                                             |
| Subprovíncia estrutural              | Grau associado à subprovíncia tectonoestrutural.                                                                          |

O grau geológico final é a **média aritmética** dessas variáveis. O documento explicita que as quatro variáveis recebem graus de 1 a 10, que os tipos de rocha componentes são ponderados por média aritmética, e que o grau final geológico é a média dos graus atribuídos às variáveis geológicas. 

Em forma de equação:

[
GEO = \frac{LIT + GEN + PROV + SUBPROV}{4}
]

onde:

[
LIT = \text{média dos graus das litologias componentes da unidade}
]

---

### 2.2 Geomorfologia

**Entrada:** base vetorial de Geomorfologia do IBGE/BDIA, escala 1:250.000.

O tema usa a quarta ordem taxonômica do mapeamento geomorfológico: os **modelados**. O IBGE distingue modelados de **acumulação (A)**, **dissecação (D)**, **dissolução (K)** e **aplanamento (P)**. Cada modelado recebe um grau de 1 a 10. 

A lógica geral é:

* modelados de **acumulação** tendem a receber notas menores;
* modelados de **aplanamento** e **dissolução** ficam em posição intermediária;
* modelados de **dissecação** tendem a receber notas maiores, principalmente quando combinam maior densidade de drenagem e maior aprofundamento das incisões.

Nos modelados de dissecação, os dois algarismos da legenda indicam **densidade de drenagem** e **aprofundamento das incisões**, variando de 1 a 5; o IBGE deu maior relevância ao aprofundamento das incisões. 

Equação operacional:

[
GEM = f(\text{modelado geomorfológico})
]

onde (f) é a tabela de correspondência dos modelados para graus de 1 a 10.

---

### 2.3 Pedologia, com o complemento do PDF 102076

**Entrada:** base vetorial de Pedologia do IBGE/BDIA, escala 1:250.000, especialmente o dado `pedo_area_[recorte]`, em formato shapefile, geometria poligonal. O complemento de 2024 detalha os campos relevantes, como `ordem`, `subordem`, `grande_gru`, `subgrupos`, `textura`, `componente`, `leg_ordem`, `legenda_2`, `cd_ord_id`, `cd_leg2_id` e área do polígono. 

O ponto essencial do complemento é que a Pedologia **não deve ser interpretada isoladamente**: ela entra junto com Geologia, Geomorfologia, Cobertura/uso, Vegetação, Declividade e Pluviosidade na álgebra final. 

A metodologia considera apenas o **solo dominante** da unidade de mapeamento. Solos subdominantes e inclusões não entram no cálculo. O complemento também esclarece que o grau pedológico representa o comportamento físico-hídrico da classe de solo em si, independentemente da classe de relevo ou da declividade. Polígonos de águas continentais e áreas urbanizadas, quando tratados como tipos de terreno no mapeamento pedológico, não são considerados para a avaliação pedológica propriamente dita. 

As três variáveis pedológicas são:

| Variável pedológica           | Papel no processo                                                                                                                          |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| Profundidade do solo          | Relacionada à infiltração, percolação e condutividade hidráulica. Solos rasos tendem a receber nota maior.                                 |
| Textura do solo               | Relacionada a areia, silte, argila e cascalhos; influencia porosidade, coesão, condutividade hidráulica, floculação e dispersão de argila. |
| Relação ou gradiente textural | Diferença de teor de argila entre horizontes superficial e subsuperficial; influencia continuidade de poros, infiltração e percolação.     |

O PDF 101684 já descreve a lógica: profundidade, textura e relação textural são escolhidas por influenciarem infiltração, condutividade hidráulica e dispersão de argila; cada atributo recebe nota de 1 a 10; a nota final é a mais restritiva, isto é, a maior nota entre os atributos avaliados. 

O complemento 102076 detalha as tabelas de atribuição. Para **profundidade**, por exemplo, solos rasos como Neossolo Litólico, Afloramentos de Rocha e Lítico recebem grau 10; Cambissolo e Saprolítico aparecem com grau 9; Argissolo e Léptico com grau 8; enquanto Latossolo e Espodossolo, muito profundos, recebem grau 1. Para **textura**, a classe siltosa recebe grau 10, combinações argilosa/siltosa e siltosa/argilosa recebem 9, e texturas orgânicas recebem os menores valores. Para **relação textural**, Abrúptico recebe 10, Argissolo 7, Argissólico e Argilúvico 6, Luvissolo 5, Planossolo 3 e Nitossólico 1.  

Equação pedológica:

[
PED = \max(PROF, TEXT, RELTEXT)
]

O uso do máximo é importante: o IBGE assume que, se uma característica do solo é muito restritiva, ela domina a avaliação pedológica.

---

### 2.4 Cobertura e uso da terra + Vegetação

**Entrada:** Monitoramento da Cobertura e Uso da Terra do Brasil 2014–2016, escala 1:1.000.000, e mapeamento de Vegetação do IBGE/BDIA. O IBGE integra Cobertura/uso e Vegetação em um único tema. Corpos d’água são excluídos da classificação de cobertura/uso para essa finalidade. 

A tabela de cobertura e uso atribui, por exemplo:

| Classe                                           | Grau |
| ------------------------------------------------ | ---: |
| Vegetação florestal                              |    1 |
| Vegetação campestre / área úmida                 |    2 |
| Silvicultura                                     |    4 |
| Área descoberta                                  |    5 |
| Mosaico de ocupações em área florestal/campestre |    6 |
| Pastagem com manejo                              |    8 |
| Área agrícola                                    |    9 |
| Área artificial                                  |   10 |

A lógica é que ambientes naturais vegetados tendem a reduzir a suscetibilidade, enquanto áreas agrícolas, pastagens e áreas artificiais elevam a potencialidade, por representarem maior antropização e alteração das condições edáficas e bióticas. 

A vegetação entra refinando as classes naturais: formações florestais, especialmente a Floresta Ombrófila Densa, recebem notas muito baixas por apresentarem maior porte, copa densa e sub-bosque; formações campestres e contatos vegetacionais recebem notas diferenciadas conforme porte, cobertura e estrutura. 

Equação operacional simplificada:

[
USOVEG = f(\text{classe de uso/cobertura}, \text{formação vegetal})
]

---

### 2.5 Declividade

**Entrada:** Modelo Digital de Elevação CGIAR-CSI-SRTM, derivado do SRTM, com pixels de 90 m x 90 m. A declividade foi calculada no ArcGIS com a função `slope`, que estima a maior inclinação entre um pixel e seus oito vizinhos. 

A declividade é o tema de maior peso no modelo final. As classes usadas foram:

| Classe         | Declividade (%) | Grau |
| -------------- | --------------: | ---: |
| Plano          |             0–3 |    1 |
| Suave-ondulado |             3–8 |    3 |
| Ondulado       |            8–20 |    5 |
| Forte-ondulado |           20–45 |    8 |
| Montanhoso     |           45–75 |    9 |
| Escarpado      |             >75 |   10 |

A premissa é direta: quanto maior a declividade, maior a probabilidade de movimentos gravitacionais de massa. 

Equação operacional:

[
DECL = f(\text{declividade percentual})
]

---

### 2.6 Pluviosidade

**Entrada:** Atlas Pluviométrico do Brasil, da CPRM, escala 1:5.000.000, usando precipitação média anual.

O IBGE reconhece que o ideal seria usar intensidade de chuva, mas, por indisponibilidade de dados homogêneos para todo o Brasil, usa a precipitação média anual como aproximação. A premissa é que áreas com maior precipitação média anual tendem, genericamente, a concentrar mais episódios intensos de chuva. 

Classes usadas:

| Precipitação média anual | Grau |
| -----------------------: | ---: |
|              400–1000 mm |    4 |
|             1000–1500 mm |    6 |
|             1500–2000 mm |    8 |
|             2000–2500 mm |    9 |
|             2500–4300 mm |   10 |

Equação operacional:

[
PLUV = f(\text{precipitação média anual})
]

---

## 3. Pseudocódigo da metodologia

a versão mais completa está apresentada em IBGE_method/IBGE_method_full_pseudocode.md

```pseudo
# ------------------------------------------------------------
# ENTRADAS
# ------------------------------------------------------------

geologia      = camada_vetorial_IBGE_BDIA_geologia_250k
geomorfologia = camada_vetorial_IBGE_BDIA_geomorfologia_250k
pedologia     = camada_vetorial_IBGE_BDIA_pedologia_250k
uso_terra     = monitoramento_cobertura_uso_terra_IBGE_1M
vegetacao     = camada_vetorial_IBGE_BDIA_vegetacao_250k
mde           = MDE_CGIAR_CSI_SRTM_90m
pluviosidade  = atlas_pluviometrico_CPRM_5M
grade_1km     = Grade_Estatistica_IBGE_1km

# ------------------------------------------------------------
# 1. ATRIBUIR GRAUS TEMÁTICOS
# ------------------------------------------------------------

# 1.1 Geologia
para cada poligono em geologia:
    litologias = poligono.litologias_componentes

    nota_litologia = media(
        lookup_litologia[litologia] para litologia em litologias
    )

    nota_genese     = lookup_genese_litologica[poligono.genese]
    nota_provincia  = lookup_provincia_estrutural[poligono.provincia]
    nota_subprov    = lookup_subprovincia_estrutural[poligono.subprovincia]

    poligono.GEO = media(
        nota_litologia,
        nota_genese,
        nota_provincia,
        nota_subprov
    )


# 1.2 Geomorfologia
para cada poligono em geomorfologia:
    modelado = poligono.modelado
    poligono.GEM = lookup_modelado_geomorfologico[modelado]


# 1.3 Pedologia
para cada poligono em pedologia:
    se poligono.tipo_terreno em ["agua continental", "area urbanizada"]:
        poligono.PED = nulo_ou_neutro
        continuar

    solo_dominante = extrair_solo_dominante(poligono.legenda)

    nota_profundidade = classificar_profundidade(
        ordem     = poligono.ordem,
        subordem  = poligono.subordem,
        grande_gru= poligono.grande_gru,
        subgrupo  = poligono.subgrupos
    )

    nota_textura = lookup_textura[poligono.textura]

    nota_relacao_textural = classificar_relacao_textural(
        ordem    = poligono.ordem,
        subgrupo = poligono.subgrupos,
        atributos_diagnosticos = poligono.legenda
    )

    # regra mais restritiva:
    poligono.PED = max(
        nota_profundidade,
        nota_textura,
        nota_relacao_textural
    )


# 1.4 Cobertura/uso da terra + Vegetação
para cada poligono em uso_terra:
    classe_uso = poligono.classe_uso

    se classe_uso em ["corpo d'agua continental", "corpo d'agua costeiro"]:
        poligono.USOVEG = nulo
        continuar

    nota_uso = lookup_uso_terra[classe_uso]

    se classe_uso representa vegetacao natural:
        formacao = buscar_formacao_vegetal_correspondente(vegetacao, poligono)
        nota_veg = lookup_formacao_vegetal[formacao]
        poligono.USOVEG = associar(nota_uso, nota_veg)
    senao:
        poligono.USOVEG = nota_uso


# 1.5 Declividade
declividade_percentual = slope(mde)

para cada pixel em declividade_percentual:
    se 0 <= pixel < 3:
        pixel.DECL = 1
    senao se 3 <= pixel < 8:
        pixel.DECL = 3
    senao se 8 <= pixel < 20:
        pixel.DECL = 5
    senao se 20 <= pixel < 45:
        pixel.DECL = 8
    senao se 45 <= pixel < 75:
        pixel.DECL = 9
    senao se pixel >= 75:
        pixel.DECL = 10


# 1.6 Pluviosidade
para cada poligono ou pixel em pluviosidade:
    p = precipitacao_media_anual_mm

    se 400 <= p < 1000:
        PLUV = 4
    senao se 1000 <= p < 1500:
        PLUV = 6
    senao se 1500 <= p < 2000:
        PLUV = 8
    senao se 2000 <= p < 2500:
        PLUV = 9
    senao se 2500 <= p <= 4300:
        PLUV = 10


# ------------------------------------------------------------
# 2. AGREGAR TODAS AS CAMADAS À GRADE DE 1 km x 1 km
# ------------------------------------------------------------

para cada celula em grade_1km:

    celula.GEO = atributo_com_maior_sobreposicao(
        geologia.GEO,
        celula
    )

    celula.GEM = atributo_com_maior_sobreposicao(
        geomorfologia.GEM,
        celula
    )

    celula.PED = atributo_com_maior_sobreposicao(
        pedologia.PED,
        celula
    )

    celula.USOVEG = atributo_com_maior_sobreposicao(
        uso_terra_vegetacao.USOVEG,
        celula
    )

    celula.DECL = atributo_com_maior_sobreposicao(
        declividade.DECL,
        celula
    )

    celula.PLUV = atributo_com_maior_sobreposicao(
        pluviosidade.PLUV,
        celula
    )


# ------------------------------------------------------------
# 3. CÁLCULO FINAL DE SUSCETIBILIDADE
# ------------------------------------------------------------

para cada celula em grade_1km:

    S = (
        0.15 * celula.GEO
      + 0.20 * celula.GEM
      + 0.15 * celula.PED
      + 0.10 * celula.USOVEG
      + 0.35 * celula.DECL
      + 0.05 * celula.PLUV
    )

    celula.SUSCETIBILIDADE_VALOR = S


# ------------------------------------------------------------
# 4. RECLASSIFICAÇÃO FINAL
# ------------------------------------------------------------

para cada celula em grade_1km:

    se S <= 3.50:
        classe = "Muito baixa"

    senao se S <= 4.50:
        classe = "Baixa"

    senao se S <= 5.50:
        classe = "Média"

    senao se S <= 6.50:
        classe = "Alta"

    senao:
        classe = "Muito alta"

    celula.SUSCETIBILIDADE_CLASSE = classe
```

---

## 4. O que o complemento 102076 muda na leitura da Pedologia

O PDF principal já dizia que a pedologia usava **profundidade, textura e relação textural**. O complemento torna isso operacional, porque explicita:

* que a base é a Pedologia vetorial do BDIA, escala 1:250.000;
* que se usa apenas o **solo dominante**;
* que subdominantes e inclusões não entram;
* que as notas pedológicas não incorporam relevo, pois relevo/declividade entram em camadas próprias;
* que profundidade, textura e gradiente/relação textural são avaliados separadamente;
* que a nota final pedológica é a **mais restritiva**, ou seja, a maior nota entre as três.

Em termos práticos, sem o 102076 você sabe a lógica geral da Pedologia; com o 102076 você consegue implementar a regra, porque ele mostra quais atributos da base pedológica devem ser lidos e como eles se traduzem em graus. 

---

## 5. Leitura crítica da metodologia

A metodologia é coerente para um produto nacional de primeira aproximação, mas tem limitações importantes:

A primeira é a **escala**. Como tudo é agregado a células de 1 km x 1 km, cada célula vira uma unidade homogênea. O próprio IBGE ressalta que percentuais de área devem ser lidos como percentuais de quadrículas, não como detalhamento intramunicipal ou de encosta específica. 

A segunda é que os pesos são **subjetivos e técnicos**, não calibrados por regressão ou aprendizado estatístico. O estudo fez simulações, eliminou resultados anômalos e escolheu a combinação considerada mais coerente pela equipe técnica. 

A terceira é que a pluviosidade usa **precipitação média anual**, não intensidade de chuva. Isso é uma aproximação por falta de dados homogêneos de intensidade para todo o Brasil. 

A quarta é que o produto final é **suscetibilidade**, não risco. Ele indica onde o meio físico e o uso/cobertura favorecem movimentos de massa, mas não calcula probabilidade temporal de ocorrência, exposição populacional, vulnerabilidade, dano esperado ou perigo em sentido estrito.

Mesmo assim, a validação apresentada é forte para a escala do produto: no gráfico de validação, 79,3% das cicatrizes de deslizamento do inventário CPRM aparecem na classe “muito alta” e 15,1% na classe “alta”, enquanto essas classes ocupam proporções bem menores da área total. Isso sugere que a álgebra de mapas concentrou corretamente a maioria das ocorrências conhecidas nas classes mais suscetíveis. 
