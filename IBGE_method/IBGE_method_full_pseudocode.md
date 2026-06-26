Abaixo está uma versão mais “implementável” do método em pseudocódigo, separando **pré-processamento**, **atribuição dos graus temáticos**, **agregação à grade de 1 km** e **cálculo final**. A lógica segue a metodologia do IBGE: temas convertidos em graus de potencialidade de 1 a 10, integração na Grade Estatística de 1 km x 1 km por máxima sobreposição e combinação final por soma ponderada. 

```pseudo
# ============================================================
# MÉTODO IBGE - SUSCETIBILIDADE A DESLIZAMENTOS DO BRASIL
# Primeira aproximação
# ============================================================

# ------------------------------------------------------------
# 0. DEFINIÇÕES GERAIS
# ------------------------------------------------------------

# Todos os temas devem ser convertidos para uma escala comum:
# grau 1  = menor potencialidade a deslizamentos
# grau 10 = maior potencialidade a deslizamentos

# Temas usados:
# GEO     = Geologia
# GEM     = Geomorfologia
# PED     = Pedologia
# USOVEG  = Cobertura e uso da terra + Vegetação
# DECL    = Declividade
# PLUV    = Pluviosidade

# Pesos finais definidos pelo IBGE:
PESO_GEO    = 0.15
PESO_GEM    = 0.20
PESO_PED    = 0.15
PESO_USOVEG = 0.10
PESO_DECL   = 0.35
PESO_PLUV   = 0.05

# Equação final:
# S = 0.15*GEO + 0.20*GEM + 0.15*PED
#   + 0.10*USOVEG + 0.35*DECL + 0.05*PLUV
```

```pseudo
# ------------------------------------------------------------
# 1. CARREGAR CAMADAS DE ENTRADA
# ------------------------------------------------------------

geologia      = carregar_vetor("BDIA_geologia_1_250000")
geomorfologia = carregar_vetor("BDIA_geomorfologia_1_250000")
pedologia     = carregar_vetor("BDIA_pedologia_1_250000")
vegetacao     = carregar_vetor("BDIA_vegetacao_1_250000")
uso_terra     = carregar_vetor("monitoramento_uso_cobertura_IBGE_1_1000000")
mde           = carregar_raster("SRTM_CGIAR_CSI_90m")
pluviosidade  = carregar_vetor_ou_raster("Atlas_Pluviometrico_CPRM_1_5000000")
grade_1km     = carregar_vetor("Grade_Estatistica_IBGE_1km")

# Padronizar sistema de referência, recorte e topologia.
para cada camada em [geologia, geomorfologia, pedologia, vegetacao,
                     uso_terra, mde, pluviosidade, grade_1km]:

    camada = reprojetar_para_SIRGAS_2000_ou_CRS_padrao(camada)
    camada = recortar_para_area_de_estudo(camada)
    camada = corrigir_geometrias_se_necessario(camada)
```

---

## 1. Geologia

O tema Geologia usa quatro variáveis: **litologia**, **característica genética da litologia**, **província estrutural** e **subprovíncia estrutural**. Para unidades com mais de uma litologia, calcula-se a média dos graus das litologias componentes. O grau final geológico é a média das variáveis geológicas. 

```pseudo
# ------------------------------------------------------------
# 2. CALCULAR GRAU GEOLÓGICO
# ------------------------------------------------------------

para cada poligono em geologia:

    # 2.1 Litologia
    litologias = extrair_litologias_componentes(poligono)

    graus_litologicos = []

    para cada litologia em litologias:
        grau_lito = tabela_grau_litologia[litologia]
        adicionar(graus_litologicos, grau_lito)

    LIT = media(graus_litologicos)

    # 2.2 Característica genética da litologia
    genese = poligono.caracteristica_genetica
    GEN = tabela_grau_genese_litologica[genese]

    # 2.3 Província estrutural
    provincia = poligono.provincia_estrutural
    PROV = tabela_grau_provincia_estrutural[provincia]

    # 2.4 Subprovíncia estrutural
    subprovincia = poligono.subprovincia_estrutural
    SUBPROV = tabela_grau_subprovincia_estrutural[subprovincia]

    # 2.5 Grau final do tema Geologia
    poligono.GEO = media(LIT, GEN, PROV, SUBPROV)
```

---

## 2. Geomorfologia

A Geomorfologia usa os **modelados geomorfológicos**, classificados de 1 a 10. Modelados de acumulação tendem a receber graus menores; modelados de dissecação, especialmente com maior aprofundamento das incisões, tendem a receber graus maiores. 

```pseudo
# ------------------------------------------------------------
# 3. CALCULAR GRAU GEOMORFOLÓGICO
# ------------------------------------------------------------

para cada poligono em geomorfologia:

    modelado = poligono.modelado

    # Ex.: Apf, Atf, Da35, Dc45, Dt54, Kc, Pgi etc.
    GEM = tabela_grau_modelado_geomorfologico[modelado]

    poligono.GEM = GEM
```

---

## 3. Pedologia, incorporando o complemento 102076

Na Pedologia, o complemento metodológico deixa claro que se usa apenas o **solo dominante** da unidade de mapeamento, ignorando solos subdominantes e inclusões. Também explicita que o grau pedológico é baseado em **profundidade**, **textura** e **gradiente/relação textural**, e que o grau final é o **mais restritivo** entre os três, isto é, o maior valor. 

```pseudo
# ------------------------------------------------------------
# 4. CALCULAR GRAU PEDOLÓGICO
# ------------------------------------------------------------

para cada poligono em pedologia:

    # 4.1 Remover classes que não entram na avaliação pedológica
    se poligono.leg_ordem em ["Água continental", "Área urbanizada"]:
        poligono.PED = NULO
        continuar

    # 4.2 Considerar apenas o solo dominante
    solo_dominante = obter_primeiro_componente_da_legenda(poligono)

    ordem      = poligono.ordem
    subordem   = poligono.subordem
    grande_gru = poligono.grande_gru
    subgrupo   = poligono.subgrupos
    textura    = poligono.textura

    # --------------------------------------------------------
    # 4.3 Grau por profundidade
    # --------------------------------------------------------

    PROF = NULO

    se ordem == "Neossolo Litólico":
        PROF = 10

    senao se poligono.tipo_terreno == "Afloramento de Rocha":
        PROF = 10

    senao se subgrupo contém "Lítico":
        PROF = 10

    senao se subgrupo contém "Saprolítico":
        PROF = 9

    senao se ordem == "Cambissolo":
        PROF = 9

    senao se subgrupo contém "Léptico":
        PROF = 8

    senao se ordem == "Argissolo":
        PROF = 8

    senao se ordem == "Luvissolo":
        PROF = 7

    senao se ordem == "Chernossolo":
        PROF = 7

    senao se ordem == "Vertissolo":
        PROF = 6

    senao se subgrupo contém "Êndico":
        PROF = 5

    senao se subgrupo contém "Arênico":
        PROF = 4

    senao se ordem == "Neossolo Regolítico":
        PROF = 4

    senao se ordem == "Neossolo Flúvico":
        PROF = 4

    senao se ordem == "Planossolo":
        PROF = 3

    senao se ordem == "Organossolo":
        PROF = 3

    senao se ordem == "Gleissolo":
        PROF = 2

    senao se ordem == "Neossolo Quartzarênico":
        PROF = 2

    senao se ordem == "Nitossolo":
        PROF = 2

    senao se subgrupo contém "Espesso":
        PROF = 2

    senao se subgrupo contém "Espessarênico":
        PROF = 2

    senao se ordem == "Plintossolo":
        PROF = 2

    senao se ordem == "Espodossolo":
        PROF = 1

    senao se ordem == "Latossolo":
        PROF = 1

    senao:
        PROF = valor_padrao_ou_revisao_tecnica


    # --------------------------------------------------------
    # 4.4 Grau por textura
    # --------------------------------------------------------

    # A tabela completa do complemento 102076 deve ser usada aqui.
    # Abaixo está a estrutura lógica.

    TEXT = tabela_grau_textura[textura]

    # Exemplos:
    # "siltosa"              -> 10
    # "argilosa/siltosa"    -> 9
    # "siltosa/argilosa"    -> 9
    # "média"               -> 3
    # "orgânica/arenosa"    -> 2
    # "orgânica"            -> 1


    # --------------------------------------------------------
    # 4.5 Grau por relação ou gradiente textural
    # --------------------------------------------------------

    RELTEXT = NULO

    se subgrupo contém "Abrúptico":
        RELTEXT = 10

    senao se ordem == "Argissolo":
        RELTEXT = 7

    senao se subgrupo contém "Argissólico":
        RELTEXT = 6

    senao se subgrupo contém "Argilúvico":
        RELTEXT = 6

    senao se ordem == "Luvissolo":
        RELTEXT = 5

    senao se subgrupo contém "Luvissólico":
        RELTEXT = 4

    senao se ordem == "Planossolo":
        RELTEXT = 3

    senao se subgrupo contém "Planossólico":
        RELTEXT = 3

    senao se ordem == "Nitossolo":
        RELTEXT = 2

    senao se subgrupo contém "Nitossólico":
        RELTEXT = 1

    senao:
        RELTEXT = valor_neutro_ou_nulo_conforme_decisao_metodologica


    # --------------------------------------------------------
    # 4.6 Grau final pedológico
    # --------------------------------------------------------

    # Regra fundamental:
    # o grau final é o mais restritivo entre profundidade,
    # textura e relação textural.

    poligono.PED = max(PROF, TEXT, RELTEXT)
```

---

## 4. Cobertura e uso da terra + Vegetação

O tema combina uso/cobertura com vegetação. Áreas mais naturais e florestadas recebem menor grau; áreas artificiais, agrícolas e pastagens tendem a receber graus maiores. Corpos d’água são excluídos dessa avaliação. 

```pseudo
# ------------------------------------------------------------
# 5. CALCULAR GRAU DE COBERTURA/USO + VEGETAÇÃO
# ------------------------------------------------------------

para cada poligono em uso_terra:

    classe_uso = poligono.classe_uso_cobertura

    se classe_uso em ["corpo d'água continental", "corpo d'água costeiro"]:
        poligono.USOVEG = NULO
        continuar

    # 5.1 Grau básico por cobertura e uso da terra
    GRAU_USO = tabela_grau_uso_cobertura[classe_uso]

    # Exemplos gerais:
    # vegetação florestal        -> grau baixo
    # vegetação campestre        -> grau baixo/intermediário
    # mosaico de ocupações       -> grau intermediário
    # pastagem                   -> grau alto
    # área agrícola              -> grau alto
    # área artificial            -> grau muito alto

    # 5.2 Refinamento por vegetação, quando aplicável
    se classe_uso representa vegetacao_natural:

        formacao_vegetal = buscar_maior_sobreposicao(
            poligono,
            vegetacao
        )

        GRAU_VEG = tabela_grau_formacao_vegetal[formacao_vegetal]

        # Dependendo da implementação, pode-se substituir ou ajustar
        # o grau de uso pelo grau específico de vegetação.
        USOVEG = combinar_grau_uso_e_vegetacao(GRAU_USO, GRAU_VEG)

    senao:

        USOVEG = GRAU_USO

    poligono.USOVEG = USOVEG
```

---

## 5. Declividade

A declividade é derivada do MDE SRTM/CGIAR-CSI. O IBGE calcula a declividade percentual e reclassifica em seis classes, com graus de 1 a 10. É o tema com maior peso no modelo final. 

```pseudo
# ------------------------------------------------------------
# 6. CALCULAR GRAU DE DECLIVIDADE
# ------------------------------------------------------------

declividade_percentual = calcular_slope_percentual(mde)

para cada pixel em declividade_percentual:

    d = pixel.valor

    se 0 <= d < 3:
        pixel.DECL = 1          # Plano

    senao se 3 <= d < 8:
        pixel.DECL = 3          # Suave-ondulado

    senao se 8 <= d < 20:
        pixel.DECL = 5          # Ondulado

    senao se 20 <= d < 45:
        pixel.DECL = 8          # Forte-ondulado

    senao se 45 <= d < 75:
        pixel.DECL = 9          # Montanhoso

    senao se d >= 75:
        pixel.DECL = 10         # Escarpado
```

---

## 6. Pluviosidade

A pluviosidade usa precipitação média anual do Atlas Pluviométrico da CPRM. O IBGE reconhece que intensidade de chuva seria mais adequada, mas usa precipitação média anual pela disponibilidade nacional homogênea. 

```pseudo
# ------------------------------------------------------------
# 7. CALCULAR GRAU DE PLUVIOSIDADE
# ------------------------------------------------------------

para cada unidade em pluviosidade:

    p = unidade.precipitacao_media_anual_mm

    se 400 <= p < 1000:
        unidade.PLUV = 4

    senao se 1000 <= p < 1500:
        unidade.PLUV = 6

    senao se 1500 <= p < 2000:
        unidade.PLUV = 8

    senao se 2000 <= p < 2500:
        unidade.PLUV = 9

    senao se 2500 <= p <= 4300:
        unidade.PLUV = 10

    senao:
        unidade.PLUV = NULO
```

---

## 7. Agregação para a grade de 1 km x 1 km

Todas as camadas são agregadas à Grade Estatística do IBGE. Quando mais de uma classe intercepta uma célula, usa-se o atributo com **maior sobreposição espacial** dentro da célula. 

```pseudo
# ------------------------------------------------------------
# 8. TRANSFERIR GRAUS TEMÁTICOS PARA A GRADE DE 1 km
# ------------------------------------------------------------

para cada celula em grade_1km:

    celula.GEO = atributo_da_classe_com_maior_area_intersectada(
        celula,
        geologia,
        campo = "GEO"
    )

    celula.GEM = atributo_da_classe_com_maior_area_intersectada(
        celula,
        geomorfologia,
        campo = "GEM"
    )

    celula.PED = atributo_da_classe_com_maior_area_intersectada(
        celula,
        pedologia,
        campo = "PED"
    )

    celula.USOVEG = atributo_da_classe_com_maior_area_intersectada(
        celula,
        uso_terra_com_vegetacao,
        campo = "USOVEG"
    )

    celula.DECL = atributo_da_classe_com_maior_area_intersectada(
        celula,
        declividade_reclassificada,
        campo = "DECL"
    )

    celula.PLUV = atributo_da_classe_com_maior_area_intersectada(
        celula,
        pluviosidade_reclassificada,
        campo = "PLUV"
    )
```

---

## 8. Cálculo final da suscetibilidade

A suscetibilidade final é uma soma ponderada. Os pesos finais foram definidos como: Geologia 15%, Geomorfologia 20%, Pedologia 15%, Cobertura/uso + Vegetação 10%, Declividade 35% e Pluviosidade 5%. 

```pseudo
# ------------------------------------------------------------
# 9. CALCULAR ÍNDICE FINAL DE SUSCETIBILIDADE
# ------------------------------------------------------------

para cada celula em grade_1km:

    se algum_dos_campos_necessarios_for_NULO(celula):
        celula.S = NULO
        celula.CLASSE = "sem dado"
        continuar

    S = (
        PESO_GEO    * celula.GEO
      + PESO_GEM    * celula.GEM
      + PESO_PED    * celula.PED
      + PESO_USOVEG * celula.USOVEG
      + PESO_DECL   * celula.DECL
      + PESO_PLUV   * celula.PLUV
    )

    celula.S = S
```

---

## 9. Reclassificação final

O valor contínuo é reclassificado em cinco classes de suscetibilidade: muito baixa, baixa, média, alta e muito alta. Os intervalos finais foram definidos por Quebras Naturais/Jenks após simulações de pesos. 

```pseudo
# ------------------------------------------------------------
# 10. RECLASSIFICAR O ÍNDICE FINAL
# ------------------------------------------------------------

para cada celula em grade_1km:

    S = celula.S

    se S é NULO:
        celula.CLASSE = "sem dado"

    senao se 0.00 <= S <= 3.50:
        celula.CLASSE = "muito baixa"

    senao se 3.50 < S <= 4.50:
        celula.CLASSE = "baixa"

    senao se 4.50 < S <= 5.50:
        celula.CLASSE = "média"

    senao se 5.50 < S <= 6.50:
        celula.CLASSE = "alta"

    senao se 6.50 < S <= 10.00:
        celula.CLASSE = "muito alta"

    senao:
        celula.CLASSE = "fora do intervalo esperado"
```

---

## 10. Fluxo completo resumido

```pseudo
# ============================================================
# FLUXO PRINCIPAL
# ============================================================

inicio:

    carregar_camadas_de_entrada()

    padronizar_CRS_recorte_e_geometrias()

    geologia.GEO = calcular_grau_geologico(
        litologia,
        genese_litologica,
        provincia_estrutural,
        subprovincia_estrutural
    )

    geomorfologia.GEM = calcular_grau_geomorfologico(
        modelado_geomorfologico
    )

    pedologia.PED = calcular_grau_pedologico(
        solo_dominante,
        profundidade,
        textura,
        relacao_textural
    )

    uso_vegetacao.USOVEG = calcular_grau_uso_vegetacao(
        cobertura_uso_terra,
        vegetacao
    )

    declividade.DECL = calcular_grau_declividade(
        slope_percentual(SRTM)
    )

    pluviosidade.PLUV = calcular_grau_pluviosidade(
        precipitacao_media_anual
    )

    grade_1km = agregar_temas_por_maior_sobreposicao(
        grade_1km,
        geologia.GEO,
        geomorfologia.GEM,
        pedologia.PED,
        uso_vegetacao.USOVEG,
        declividade.DECL,
        pluviosidade.PLUV
    )

    para cada celula em grade_1km:

        celula.S = (
            0.15 * celula.GEO
          + 0.20 * celula.GEM
          + 0.15 * celula.PED
          + 0.10 * celula.USOVEG
          + 0.35 * celula.DECL
          + 0.05 * celula.PLUV
        )

        celula.CLASSE = reclassificar_suscetibilidade(celula.S)

    exportar_mapa(
        grade_1km,
        campos = ["GEO", "GEM", "PED", "USOVEG", "DECL", "PLUV", "S", "CLASSE"]
    )

fim
```

A essência operacional é esta: **tabelas de correspondência transformam atributos ambientais em notas de 1 a 10; as notas são transferidas para uma grade comum; a grade recebe uma média ponderada; o resultado é reclassificado em cinco classes finais**.
