"""Build the Portuguese static report for the IBGE high-resolution product."""

from __future__ import annotations

import html
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import rasterio
from matplotlib import colormaps
from PIL import Image
from rasterio.enums import Resampling
from rasterio.warp import transform

from src.three_method_comparison import (
    IBGE_WEIGHTS,
    ROOT,
    SIG_PATHS,
    custom_lulc_land_use_mapping,
)


OUTPUTS = ROOT / "IBGE_method" / "outputs"
REPORTS = ROOT / "IBGE_method" / "reports"
CONFIGS = ROOT / "IBGE_method" / "configs"
WEBSITE = ROOT / "IBGE_method" / "website"
ASSETS = WEBSITE / "assets"
FINAL_DTM = Path(os.environ.get("IBGE_DTM_PATH", "/home/kaue/data/landslide/dtm_final.tif")).expanduser().resolve()

THEME_RASTERS = {
    "dtm": FINAL_DTM,
    "lulc": OUTPUTS / "ibge_land_use_custom_lulc.tif",
    "slope": OUTPUTS / "ibge_note_slope.tif",
    "geology": OUTPUTS / "ibge_note_geology.tif",
    "pedology": OUTPUTS / "ibge_note_pedology.tif",
    "geomorphology": OUTPUTS / "ibge_note_geomorphology.tif",
    "pluviosity": OUTPUTS / "ibge_note_pluviosity.tif",
    "pluviosity_mm": OUTPUTS / "ibge_pluviosity_pma_mm.tif",
    "score": OUTPUTS / "ibge_susceptibility_score_1to10.tif",
    "class5": OUTPUTS / "ibge_class_map_5class.tif",
    "valid": OUTPUTS / "ibge_valid_mask.tif",
}

DISCRETE_COLORS = {
    "lulc": {
        0: (245, 247, 250),
        1: (188, 32, 111),
        2: (210, 180, 91),
        3: (64, 135, 194),
        4: (133, 180, 84),
        5: (41, 118, 80),
    },
    "class5": {
        1: (47, 111, 166),
        2: (77, 163, 121),
        3: (232, 196, 92),
        4: (218, 130, 72),
        5: (174, 64, 67),
        255: (232, 235, 238),
    },
    "valid": {
        0: (232, 235, 238),
        1: (53, 128, 88),
    },
}

LULC_NAMES = {
    1: "Area artificial",
    2: "Area descoberta",
    3: "Corpo d'agua",
    4: "Vegetacao campestre",
    5: "Vegetacao florestal",
}


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def rel(path: Path) -> str:
    return path.relative_to(WEBSITE).as_posix()


def raster_sample(
    path: Path,
    *,
    max_size: int = 1300,
    nearest: bool = False,
    mask_path: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    with rasterio.open(path) as src:
        scale = max(src.width / max_size, src.height / max_size, 1.0)
        width = max(1, int(round(src.width / scale)))
        height = max(1, int(round(src.height / scale)))
        data = src.read(
            1,
            out_shape=(height, width),
            masked=True,
            resampling=Resampling.nearest if nearest else Resampling.bilinear,
        )
    arr = np.asarray(data.astype(np.float32).filled(np.nan), dtype=np.float32)
    mask = ~np.ma.getmaskarray(data)
    if mask_path is not None:
        with rasterio.open(mask_path) as mask_src:
            mask_data = mask_src.read(
                1,
                out_shape=arr.shape,
                masked=True,
                resampling=Resampling.nearest,
            )
            mask_nodata = mask_src.nodata
        mask_arr = np.asarray(mask_data.astype(np.float32).filled(np.nan), dtype=np.float32)
        mask_valid = ~np.ma.getmaskarray(mask_data) & np.isfinite(mask_arr)
        if mask_nodata is not None:
            mask_valid &= mask_arr != float(mask_nodata)
        mask &= mask_valid
    return arr, mask


def continuous_rgb(
    data: np.ndarray,
    mask: np.ndarray,
    *,
    cmap_name: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> np.ndarray:
    valid = mask & np.isfinite(data)
    if vmin is None or vmax is None:
        if np.any(valid):
            lo, hi = np.percentile(data[valid], [2, 98])
            vmin = float(lo if vmin is None else vmin)
            vmax = float(hi if vmax is None else vmax)
        else:
            vmin, vmax = 0.0, 1.0
    if math.isclose(float(vmin), float(vmax)):
        vmax = float(vmin) + 1.0
    scaled = np.clip((data - float(vmin)) / (float(vmax) - float(vmin)), 0.0, 1.0)
    rgba = colormaps[cmap_name](np.nan_to_num(scaled, nan=0.0))
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    rgb[~valid] = np.array([235, 238, 241], dtype=np.uint8)
    return rgb


def discrete_rgb(data: np.ndarray, mask: np.ndarray, palette: Mapping[int, Tuple[int, int, int]]) -> np.ndarray:
    rgb = np.full((*data.shape, 3), 235, dtype=np.uint8)
    rounded = np.rint(np.nan_to_num(data, nan=-999999.0)).astype(np.int32)
    for value, color in palette.items():
        rgb[(rounded == int(value)) & mask] = np.array(color, dtype=np.uint8)
    rgb[~mask] = np.array([235, 238, 241], dtype=np.uint8)
    return rgb


def save_jpeg(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path, "JPEG", quality=88, optimize=True)


def save_png(path: Path, rgba: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba).save(path, "PNG", optimize=True)


def make_thumbnails() -> Dict[str, str]:
    outputs: Dict[str, str] = {}
    specs = {
        "dtm": ("thumb_dtm.jpg", "gray", None, None, False),
        "slope": ("thumb_slope.jpg", "magma", 0.0, 10.0, False),
        "geology": ("thumb_geology.jpg", "viridis", 0.0, 10.0, True),
        "pedology": ("thumb_pedology.jpg", "viridis", 0.0, 10.0, True),
        "geomorphology": ("thumb_geomorphology.jpg", "viridis", 0.0, 10.0, True),
        "pluviosity": ("thumb_pluviosity.jpg", "viridis", 0.0, 10.0, False),
        "pluviosity_mm": ("thumb_pluviosity_mm.jpg", "Blues", None, None, False),
        "score": ("thumb_score.jpg", "viridis", 0.0, 10.0, False),
    }
    for key, (filename, cmap, vmin, vmax, nearest) in specs.items():
        data, mask = raster_sample(THEME_RASTERS[key], nearest=nearest)
        out = ASSETS / filename
        save_jpeg(out, continuous_rgb(data, mask, cmap_name=cmap, vmin=vmin, vmax=vmax))
        outputs[key] = rel(out)

    for key, filename in [("lulc", "thumb_lulc.jpg"), ("class5", "thumb_class5.jpg"), ("valid", "thumb_valid.jpg")]:
        data, mask = raster_sample(
            THEME_RASTERS[key],
            nearest=True,
            mask_path=FINAL_DTM if key == "lulc" else None,
        )
        out = ASSETS / filename
        save_jpeg(out, discrete_rgb(data, mask, DISCRETE_COLORS[key]))
        outputs[key] = rel(out)

    outputs["overview"] = outputs["score"]
    return outputs


def make_webviewer_asset() -> Dict[str, Any]:
    data, mask = raster_sample(THEME_RASTERS["score"], max_size=1800, nearest=False)
    valid = mask & np.isfinite(data) & (data != -9999)
    rgb = continuous_rgb(data, valid, cmap_name="viridis", vmin=0.0, vmax=10.0)
    alpha = np.where(valid, 215, 0).astype(np.uint8)
    rgba = np.dstack([rgb, alpha])
    image_path = ASSETS / "webviewer_score.png"
    save_png(image_path, rgba)

    with rasterio.open(THEME_RASTERS["score"]) as src:
        bounds = src.bounds
        xs = [bounds.left, bounds.right, bounds.right, bounds.left]
        ys = [bounds.top, bounds.top, bounds.bottom, bounds.bottom]
        lon, lat = transform(src.crs, "EPSG:4326", xs, ys)
    coords = [[float(x), float(y)] for x, y in zip(lon, lat)]
    meta = {
        "image": rel(image_path),
        "coordinates": coords,
        "bounds": [
            [min(c[0] for c in coords), min(c[1] for c in coords)],
            [max(c[0] for c in coords), max(c[1] for c in coords)],
        ],
        "center": [
            sum(c[0] for c in coords) / 4.0,
            sum(c[1] for c in coords) / 4.0,
        ],
    }
    (ASSETS / "webviewer_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def raster_stats(
    path: Path,
    *,
    discrete: bool = False,
    include_nodata: bool = False,
    mask_path: Optional[Path] = None,
) -> Dict[str, Any]:
    with rasterio.open(path) as src:
        mask_src = rasterio.open(mask_path) if mask_path is not None else None
        pixel_area = abs(float(src.transform.a * src.transform.e))
        counts: Counter[int] = Counter()
        total = 0
        total_sum = 0.0
        total_sumsq = 0.0
        min_value = math.inf
        max_value = -math.inf
        try:
            for _, window in src.block_windows(1):
                arr = src.read(1, window=window)
                valid = np.isfinite(arr)
                if src.nodata is not None and not include_nodata:
                    valid &= arr != src.nodata
                if mask_src is not None:
                    mask_arr = mask_src.read(1, window=window)
                    mask_valid = np.isfinite(mask_arr)
                    if mask_src.nodata is not None:
                        mask_valid &= mask_arr != mask_src.nodata
                    valid &= mask_valid
                vals = arr[valid]
                if vals.size == 0:
                    continue
                if discrete:
                    unique, freq = np.unique(vals.astype(np.int64), return_counts=True)
                    for value, count in zip(unique, freq):
                        counts[int(value)] += int(count)
                else:
                    vals64 = vals.astype(np.float64)
                    total += int(vals64.size)
                    total_sum += float(vals64.sum())
                    total_sumsq += float(np.square(vals64).sum())
                    min_value = min(min_value, float(vals64.min()))
                    max_value = max(max_value, float(vals64.max()))
        finally:
            if mask_src is not None:
                mask_src.close()
        if discrete:
            total_count = sum(counts.values())
            return {
                "values": {
                    str(value): {
                        "pixels": count,
                        "area_m2": count * pixel_area,
                        "fraction": count / total_count if total_count else 0.0,
                    }
                    for value, count in sorted(counts.items())
                },
                "total_pixels": total_count,
            }
        mean = total_sum / total if total else 0.0
        variance = max(total_sumsq / total - mean * mean, 0.0) if total else 0.0
        return {
            "min": min_value if total else None,
            "max": max_value if total else None,
            "mean": mean,
            "std": math.sqrt(variance),
            "pixels": total,
        }


def mapping_table(rows: Iterable[Sequence[Any]], headers: Sequence[str]) -> str:
    head = "".join(f"<th>{escape(header)}</th>" for header in headers)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{escape(cell)}</td>" for cell in row) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def stats_table(stats: Mapping[str, Any], *, labels: Optional[Mapping[int, str]] = None) -> str:
    if "values" in stats:
        rows = []
        for value, payload in stats["values"].items():
            label = labels.get(int(value), "") if labels else ""
            rows.append(
                [
                    value,
                    label,
                    payload["pixels"],
                    f"{payload['area_m2']:.2f}",
                    f"{100.0 * payload['fraction']:.2f}%",
                ]
            )
        return mapping_table(rows, ["Valor", "Classe", "Pixels", "Area (m2)", "Fração"])
    return mapping_table(
        [
            ["Mínimo", f"{stats.get('min'):.4f}" if stats.get("min") is not None else "sem dado"],
            ["Máximo", f"{stats.get('max'):.4f}" if stats.get("max") is not None else "sem dado"],
            ["Média", f"{stats.get('mean'):.4f}"],
            ["Desvio padrão", f"{stats.get('std'):.4f}"],
            ["Pixels válidos", stats.get("pixels")],
        ],
        ["Métrica", "Valor"],
    )


def section(title: str, image: str, text: str, *tables: str) -> str:
    return (
        f"<div class=\"figure\"><img src=\"{escape(image)}\" alt=\"{escape(title)}\"></div>"
        f"<div class=\"copy\"><h2>{escape(title)}</h2>{text}{''.join(tables)}</div>"
    )


def build_html(config: Mapping[str, Any], summary: Mapping[str, Any], thumbs: Mapping[str, str], viewer: Mapping[str, Any], stats: Mapping[str, Any]) -> str:
    lulc_mapping = custom_lulc_land_use_mapping()
    lulc_rows = [[k, LULC_NAMES.get(k, ""), v] for k, v in sorted(lulc_mapping.items())]
    slope_rows = [[lo, hi if hi is not None else ">=75", note] for lo, hi, note in config["slope_classes_percent_to_note"]]
    pluv_rows = [
        ["400-1000", 4],
        ["1000-1500", 6],
        ["1500-2000", 8],
        ["2000-2500", 9],
        ["2500-4300", 10],
        ["adaptação 16 cm", "interpolação contínua entre os limiares acima"],
    ]
    geom_rows = sorted(config["input_proxies"]["GEM"]["mapped_values"].items())
    geo_rows = sorted(config["input_proxies"]["GEO"]["mapped_values"].items())
    ped_rows = sorted(config["input_proxies"]["PED"]["mapped_values"].items())
    weights_rows = [[key, value] for key, value in IBGE_WEIGHTS.items()]

    tabs = [
        ("overview", "Visão Geral"),
        ("inputs", "Entradas"),
        ("lulc", "LULC / USOVEG"),
        ("slope", "Declividade"),
        ("geology", "Geologia"),
        ("pedology", "Pedologia"),
        ("geomorphology", "Geomorfologia"),
        ("pluviosity", "Pluviosidade"),
        ("algebra", "Álgebra IBGE"),
        ("results", "Resultados"),
        ("webviewer", "webviewer"),
    ]
    nav = "".join(
        f"<button class=\"tab-button{' active' if idx == 0 else ''}\" data-tab=\"{tab_id}\">{label}</button>"
        for idx, (tab_id, label) in enumerate(tabs)
    )

    tab_content = {
        "overview": section(
            "Produto IBGE adaptado em 16 cm",
            thumbs["overview"],
            (
                "<p>Este relatório documenta a geração do produto IBGE adaptado de alta resolução. "
                "A grade estatística nacional de 1 km foi substituída pelo grid do DTM final em 16 cm; "
                "os pesos, a escala de notas e as classes finais da metodologia IBGE foram preservados.</p>"
                f"<p><strong>Fração válida:</strong> {summary['valid_fraction']:.6f}. "
                f"<strong>Score 1-10:</strong> {summary['score_1_to_10_min']:.4f} a {summary['score_1_to_10_max']:.4f}.</p>"
            ),
            mapping_table(weights_rows, ["Tema", "Peso"]),
        ),
        "inputs": section(
            "Entradas e proxies",
            thumbs["dtm"],
            (
                f"<p><strong>DTM:</strong> {escape(config['input_proxies']['DECL']['path'])}</p>"
                f"<p><strong>LULC:</strong> {escape(config['input_proxies']['USOVEG'].get('path', ''))}</p>"
                f"<p><strong>Geologia:</strong> {escape(SIG_PATHS['geology'])}</p>"
                f"<p><strong>Pedologia:</strong> {escape(SIG_PATHS['pedology'])}</p>"
                f"<p><strong>Geomorfologia:</strong> {escape(SIG_PATHS['relief'])}</p>"
                f"<p><strong>Pluviosidade:</strong> {escape(SIG_PATHS['rain_pma2'])}</p>"
            ),
        ),
        "lulc": section(
            "LULC / USOVEG",
            thumbs["lulc"],
            "<p>Fonte: ensemble LULC full-res gerado por deep learning. A classe de água recebe nota 0 e é excluída dos pixels válidos.</p>",
            mapping_table(lulc_rows, ["Código LULC", "Classe", "Nota IBGE"]),
            stats_table(stats["lulc"], labels=LULC_NAMES),
        ),
        "slope": section(
            "Declividade",
            thumbs["slope"],
            "<p>Fonte: DTM final do drone. A declividade percentual é calculada no grid de 16 cm com halo de 1 pixel nos tiles.</p>",
            mapping_table(slope_rows, ["Declividade (%) - início", "Declividade (%) - fim", "Nota IBGE"]),
            stats_table(stats["slope"]),
        ),
        "geology": section(
            "Geologia",
            thumbs["geology"],
            (
                "<p>Proxy adaptado: a metodologia IBGE estrita calcula GEO a partir da média de "
                "litologia, gênese, província estrutural e subprovíncia estrutural. Nesta versão "
                "de alta resolução não há, no recorte operacional, a tabela BDIA completa com esses "
                "quatro componentes normalizados para cada polígono. Por isso, a camada local de "
                "geologia é usada como proxy compatível.</p>"
                "<p>O campo <code>SIGLA_UNID</code> identifica a unidade geológica dominante. Cada "
                "unidade foi convertida diretamente para uma nota GEO 1-10 já interpretada para "
                "potencialidade a movimentos de massa. No recorte atual, apenas a unidade "
                "<code>PRps</code> intercepta a área válida do DTM, resultando em nota geológica "
                "espacialmente constante. Isso não indica falha de rasterização; indica limitação "
                "da variabilidade geológica disponível dentro desta área de estudo.</p>"
            ),
            mapping_table(geo_rows, ["SIGLA_UNID", "Nota GEO"]),
            stats_table(stats["geology"]),
        ),
        "pedology": section(
            "Pedologia",
            thumbs["pedology"],
            (
                "<p>Proxy adaptado: na metodologia IBGE estrita, PED é calculado como "
                "<code>max(PROF, TEXT, RELTEXT)</code>, considerando o solo dominante e atributos "
                "pedológicos como profundidade, textura e relação/gradiente textural. A camada "
                "local disponível para esta área não expõe todos esses campos analíticos de forma "
                "compatível com a tabela BDIA/pedologia completa.</p>"
                "<p>Assim, o campo <code>DESC_</code> foi usado como legenda pedológica operacional. "
                "Cada legenda recebeu uma nota PED proxy. No recorte válido, a classe dominante é "
                "<code>PEe1</code>, mapeada para nota 8, enquanto <code>Rios</code> recebe 0 e é "
                "excluído da máscara válida. Portanto, a baixa variação pedológica no produto final "
                "decorre da resolução temática e da legenda disponível, não de simplificação no grid "
                "de 16 cm.</p>"
            ),
            mapping_table(ped_rows, ["DESC_", "Nota PED"]),
            stats_table(stats["pedology"]),
        ),
        "geomorphology": section(
            "Geomorfologia",
            thumbs["geomorphology"],
            (
                "<p>Proxy adaptado: a metodologia IBGE estrita usa modelados geomorfológicos BDIA, "
                "especialmente a quarta ordem taxonômica, para diferenciar acumulação, dissecação, "
                "dissolução e aplanamento. Nesta adaptação, a camada local de padrões de relevo foi "
                "usada como substituta de maior compatibilidade espacial com o estudo.</p>"
                "<p>O campo <code>Classe</code> representa formas de relevo como planícies, colinas, "
                "morros baixos e morros altos. A transformação segue a lógica geomorfológica do IBGE: "
                "ambientes deposicionais/planos recebem notas menores, formas intermediárias recebem "
                "notas médias e relevo mais dissecado/íngreme recebe notas altas. Diferentemente da "
                "geologia e da pedologia, esta camada apresenta variação espacial relevante dentro "
                "da área, contribuindo para contrastes locais no score final.</p>"
            ),
            mapping_table(geom_rows, ["Classe", "Nota GEM"]),
            stats_table(stats["geomorphology"]),
        ),
        "pluviosity": section(
            "Pluviosidade",
            thumbs["pluviosity"],
            "<p>Fonte: Atlas Pluviométrico CPRM/SGB PMA. Para a adaptação 16 cm, a nota é interpolada continuamente entre os limiares IBGE para preservar a variação local.</p>",
            mapping_table(pluv_rows, ["Precipitação média anual (mm)", "Nota PLUV"]),
            stats_table(stats["pluviosity"]),
            f"<h3>PMA em mm</h3>{stats_table(stats['pluviosity_mm'])}<div class=\"figure inline\"><img src=\"{thumbs['pluviosity_mm']}\" alt=\"PMA em mm\"></div>",
        ),
        "algebra": section(
            "Álgebra IBGE",
            thumbs["score"],
            (
                "<p>O score final é calculado por pixel:</p>"
                "<p><code>S = 0.15*GEO + 0.20*GEM + 0.15*PED + 0.10*USOVEG + 0.35*DECL + 0.05*PLUV</code></p>"
                "<p>Pixels com água, nodata ou nota temática inválida são excluídos da máscara válida.</p>"
            ),
            mapping_table(weights_rows, ["Tema", "Peso"]),
        ),
        "results": section(
            "Resultados",
            thumbs["class5"],
            (
                "<p>O score 1-10 é reclassificado em cinco classes: muito baixa, baixa, média, alta e muito alta.</p>"
                f"<p><strong>Score 1-10:</strong> {summary['score_1_to_10_min']:.4f} a {summary['score_1_to_10_max']:.4f}.</p>"
            ),
            stats_table(stats["class5"], labels={1: "Muito baixa", 2: "Baixa", 3: "Média", 4: "Alta", 5: "Muito alta", 255: "Sem dado"}),
            f"<h3>Máscara válida</h3>{stats_table(stats['valid'], labels={0: 'Inválido', 1: 'Válido'})}<div class=\"figure inline\"><img src=\"{thumbs['valid']}\" alt=\"Máscara válida\"></div>",
        ),
        "webviewer": (
            "<div class=\"viewer-shell\">"
            "<aside class=\"viewer-panel\">"
            "<h2>webviewer</h2>"
            "<p>Score final 0-10 sobreposto ao mapa base. A imagem foi derivada de "
            "<code>ibge_susceptibility_score_1to10.tif</code> e está registrada em WGS84.</p>"
            "<h3>Mapa de fundo</h3>"
            "<div class=\"map-controls\" role=\"group\" aria-label=\"Mapa de fundo\">"
            "<button type=\"button\" class=\"base-button active\" data-base=\"osm\">OSM</button>"
            "<button type=\"button\" class=\"base-button\" data-base=\"bing\">Bing imagem</button>"
            "</div>"
            "<h3>Camada IBGE</h3>"
            "<label class=\"opacity-control\">Opacidade do score"
            "<input id=\"score-opacity\" type=\"range\" min=\"0\" max=\"1\" step=\"0.05\" value=\"0.88\">"
            "</label>"
            "<div class=\"legend\"><span>0</span><div class=\"ramp\"></div><span>10</span></div>"
            "<p class=\"viewer-note\">Viridis: valores menores em roxo/azul e maiores em verde/amarelo.</p>"
            "</aside>"
            "<div class=\"map-wrap\"><div id=\"map\"></div></div>"
            "</div>"
        ),
    }
    sections = "".join(
        f"<section id=\"{tab_id}\" class=\"tab-panel{' active' if idx == 0 else ''}\">{tab_content[tab_id]}</section>"
        for idx, (tab_id, _) in enumerate(tabs)
    )

    viewer_json = json.dumps(viewer, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Produto IBGE Adaptado 16 cm</title>
  <link href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css" rel="stylesheet">
  <style>
    :root {{ --ink:#18212b; --muted:#5b6673; --line:#d8dee6; --bg:#f6f8fb; --panel:#ffffff; --accent:#1f6f8b; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color:var(--ink); background:var(--bg); }}
    header {{ padding:24px 32px 16px; background:#0f2430; color:white; }}
    header h1 {{ margin:0 0 8px; font-size:28px; letter-spacing:0; }}
    header p {{ margin:0; color:#d5e4ec; max-width:1050px; line-height:1.5; }}
    nav {{ display:flex; flex-wrap:wrap; gap:6px; padding:12px 32px; background:#e9eef4; border-bottom:1px solid var(--line); position:sticky; top:0; z-index:5; }}
    button.tab-button {{ border:1px solid #c8d2dc; background:white; color:#1b2a36; padding:9px 12px; border-radius:6px; cursor:pointer; font-size:14px; }}
    button.tab-button.active {{ background:var(--accent); color:white; border-color:var(--accent); }}
    .map-controls {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; margin:10px 0 12px; }}
    .base-button {{ border:1px solid #c8d2dc; background:white; color:#1b2a36; padding:8px 12px; border-radius:6px; cursor:pointer; font-size:14px; }}
    .base-button.active {{ background:#263b4a; color:white; border-color:#263b4a; }}
    main {{ padding:24px 32px 40px; }}
    .tab-panel {{ display:none; grid-template-columns:minmax(280px, 42%) minmax(320px, 1fr); gap:24px; align-items:start; }}
    .tab-panel.active {{ display:grid; }}
    .figure img {{ width:100%; max-height:720px; object-fit:contain; background:white; border:1px solid var(--line); border-radius:8px; }}
    .figure.inline img {{ max-width:680px; margin-top:12px; }}
    .copy, #webviewer {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:20px; }}
    h2 {{ margin:0 0 12px; font-size:22px; }}
    h3 {{ margin:20px 0 8px; font-size:17px; }}
    p {{ line-height:1.55; color:var(--muted); }}
    code {{ background:#eef2f6; padding:2px 5px; border-radius:4px; }}
    table {{ width:100%; border-collapse:collapse; margin:14px 0 18px; font-size:14px; }}
    th, td {{ text-align:left; border-bottom:1px solid var(--line); padding:8px 10px; vertical-align:top; }}
    th {{ background:#f0f4f8; color:#22313d; }}
    #webviewer {{ grid-column:1 / -1; background:transparent; border:0; padding:0; }}
    .viewer-shell {{ display:grid; grid-template-columns:320px minmax(420px,1fr); gap:16px; align-items:stretch; }}
    .viewer-panel {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:18px; }}
    .viewer-panel p {{ margin-top:0; }}
    .viewer-note {{ font-size:13px; }}
    .opacity-control {{ display:grid; gap:8px; color:var(--muted); font-size:14px; margin:8px 0 14px; }}
    .opacity-control input {{ width:100%; }}
    .map-wrap {{ background:white; border:1px solid var(--line); border-radius:8px; padding:8px; }}
    #map {{ width:100%; height:76vh; min-height:560px; border-radius:6px; }}
    .legend {{ display:flex; align-items:center; gap:10px; margin-top:10px; color:var(--muted); }}
    .ramp {{ height:14px; width:280px; border-radius:999px; border:1px solid #c5cbd3; background:linear-gradient(90deg,#440154,#3b528b,#21918c,#5ec962,#fde725); }}
    @media (max-width:900px) {{ .tab-panel.active {{ display:block; }} .copy {{ margin-top:18px; }} .viewer-shell {{ display:block; }} .map-wrap {{ margin-top:14px; }} nav, main, header {{ padding-left:16px; padding-right:16px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Produto IBGE adaptado em grade de 16 cm</h1>
    <p>Relatório técnico em português da geração do produto final: entradas, proxies, mapeamentos, álgebra e visualização interativa do score de suscetibilidade.</p>
  </header>
  <nav>{nav}</nav>
  <main>{sections}</main>
  <script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
  <script>
    document.querySelectorAll('.tab-button').forEach((button) => {{
      button.addEventListener('click', () => {{
        document.querySelectorAll('.tab-button').forEach((b) => b.classList.remove('active'));
        document.querySelectorAll('.tab-panel').forEach((p) => p.classList.remove('active'));
        button.classList.add('active');
        document.getElementById(button.dataset.tab).classList.add('active');
        if (button.dataset.tab === 'webviewer' && window.ibgeMap) {{
          setTimeout(() => window.ibgeMap.resize(), 50);
        }}
      }});
    }});
    const viewer = {viewer_json};
    window.ibgeMap = new maplibregl.Map({{
      container: 'map',
      style: {{
        version: 8,
        sources: {{
          osm: {{
            type: 'raster',
            tiles: ['https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png'],
            tileSize: 256,
            attribution: '© OpenStreetMap contributors'
          }},
          bing: {{
            type: 'raster',
            tiles: [
              'https://ecn.t0.tiles.virtualearth.net/tiles/a{{quadkey}}.jpeg?g=129&mkt=pt-BR&n=z',
              'https://ecn.t1.tiles.virtualearth.net/tiles/a{{quadkey}}.jpeg?g=129&mkt=pt-BR&n=z',
              'https://ecn.t2.tiles.virtualearth.net/tiles/a{{quadkey}}.jpeg?g=129&mkt=pt-BR&n=z',
              'https://ecn.t3.tiles.virtualearth.net/tiles/a{{quadkey}}.jpeg?g=129&mkt=pt-BR&n=z'
            ],
            tileSize: 256,
            attribution: '© Microsoft Bing'
          }}
        }},
        layers: [
          {{ id: 'osm-base', type: 'raster', source: 'osm', layout: {{ visibility: 'visible' }} }},
          {{ id: 'bing-base', type: 'raster', source: 'bing', layout: {{ visibility: 'none' }} }}
        ]
      }},
      center: viewer.center,
      zoom: 14,
      attributionControl: false
    }});
    window.ibgeMap.addControl(new maplibregl.NavigationControl({{ visualizePitch: false }}), 'top-right');
    window.ibgeMap.on('load', () => {{
      window.ibgeMap.addSource('score', {{
        type: 'image',
        url: viewer.image,
        coordinates: viewer.coordinates
      }});
      window.ibgeMap.addLayer({{
        id: 'score-layer',
        type: 'raster',
        source: 'score',
        paint: {{ 'raster-opacity': 0.88 }}
      }});
      window.ibgeMap.fitBounds(viewer.bounds, {{ padding: 30, duration: 0 }});
    }});
    document.getElementById('score-opacity').addEventListener('input', (event) => {{
      window.ibgeMap.setPaintProperty('score-layer', 'raster-opacity', Number(event.target.value));
    }});
    document.querySelectorAll('.base-button').forEach((button) => {{
      button.addEventListener('click', () => {{
        document.querySelectorAll('.base-button').forEach((b) => b.classList.remove('active'));
        button.classList.add('active');
        const useOsm = button.dataset.base === 'osm';
        window.ibgeMap.setLayoutProperty('osm-base', 'visibility', useOsm ? 'visible' : 'none');
        window.ibgeMap.setLayoutProperty('bing-base', 'visibility', useOsm ? 'none' : 'visible');
      }});
    }});
  </script>
</body>
</html>
"""


def main() -> int:
    ASSETS.mkdir(parents=True, exist_ok=True)
    config = read_json(CONFIGS / "method_config.json")
    summary = read_json(REPORTS / "summary.json")
    thumbnails = make_thumbnails()
    viewer = make_webviewer_asset()
    active_dtm = FINAL_DTM
    stats = {
        "lulc": raster_stats(THEME_RASTERS["lulc"], discrete=True, mask_path=active_dtm),
        "slope": raster_stats(THEME_RASTERS["slope"], mask_path=active_dtm),
        "geology": raster_stats(THEME_RASTERS["geology"], mask_path=active_dtm),
        "pedology": raster_stats(THEME_RASTERS["pedology"], mask_path=active_dtm),
        "geomorphology": raster_stats(THEME_RASTERS["geomorphology"], mask_path=active_dtm),
        "pluviosity": raster_stats(THEME_RASTERS["pluviosity"], mask_path=active_dtm),
        "pluviosity_mm": raster_stats(THEME_RASTERS["pluviosity_mm"], mask_path=active_dtm),
        "class5": raster_stats(THEME_RASTERS["class5"], discrete=True),
        "valid": raster_stats(THEME_RASTERS["valid"], discrete=True, include_nodata=True),
    }
    report_data = {
        "config": config,
        "summary": summary,
        "thumbnails": thumbnails,
        "webviewer": viewer,
        "stats": stats,
    }
    (ASSETS / "report_data.json").write_text(json.dumps(report_data, indent=2, ensure_ascii=False), encoding="utf-8")
    (WEBSITE / "index.html").write_text(build_html(config, summary, thumbnails, viewer, stats), encoding="utf-8")
    print(f"[ibge-website] Wrote {WEBSITE / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
