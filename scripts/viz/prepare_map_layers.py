"""Convert result rasters into web-ready map layers for the interactive app.

Each GeoTIFF (EPSG:32645, 30 m) is reprojected to WGS84 (EPSG:4326), coloured and written as a
transparent PNG plus a JSON file with its corner coordinates. The app then only overlays these
images on a basemap and does not need rasterio itself.

    python scripts/viz/prepare_map_layers.py      # needs rasterio, matplotlib, pillow
"""
import json
from pathlib import Path

import numpy as np
import rasterio
from matplotlib import colormaps
from PIL import Image
from rasterio.warp import Resampling, calculate_default_transform, reproject

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results_v2"
OUT = ROOT / "app" / "assets" / "maps"
DST_CRS = "EPSG:4326"
# Output pixels per side relative to the 30 m source; 2 gives ~60 m, plenty for a web map and ~4x smaller files
DOWNSCALE = 2


def reproject_to_wgs84(path: Path) -> tuple[np.ndarray, list[list[float]]]:
    """Return the raster in EPSG:4326 (NaN outside the data) and its corners, clockwise from top-left."""
    with rasterio.open(path) as src:
        _, width, height = calculate_default_transform(src.crs, DST_CRS, src.width, src.height, *src.bounds)
        transform, width, height = calculate_default_transform(
            src.crs, DST_CRS, src.width, src.height, *src.bounds,
            dst_width=width // DOWNSCALE, dst_height=height // DOWNSCALE)
        band = src.read(1).astype("float32")
        if src.nodata is not None:
            band[band == src.nodata] = np.nan
        out = np.full((height, width), np.nan, dtype="float32")
        reproject(band, out, src_transform=src.transform, src_crs=src.crs, src_nodata=np.nan,
                  dst_transform=transform, dst_crs=DST_CRS, dst_nodata=np.nan,
                  resampling=Resampling.nearest)
    west, north = transform * (0, 0)
    east, south = transform * (width, height)
    return out, [[west, north], [east, north], [east, south], [west, south]]


def colour_continuous(values: np.ndarray, cmap: str, vmin: float, vmax: float) -> np.ndarray:
    """Map values to RGBA with a matplotlib colormap; NaN becomes fully transparent."""
    scaled = np.clip((values - vmin) / (vmax - vmin), 0, 1)
    rgba = (colormaps[cmap](np.nan_to_num(scaled)) * 255).astype("uint8")
    rgba[np.isnan(values), 3] = 0
    return rgba


def colour_classes(values: np.ndarray, palette: dict[int, str]) -> np.ndarray:
    """Map integer class codes to RGBA hex colours (#RRGGBB or #RRGGBBAA); other values become transparent."""
    rgba = np.zeros(values.shape + (4,), dtype="uint8")
    for code, colour in palette.items():
        colour = colour.lstrip("#") + ("ff" if len(colour) == 7 else "")
        rgba[values == code] = [int(colour[i:i + 2], 16) for i in (0, 2, 4, 6)]
    return rgba


def write_layer(name: str, rgba: np.ndarray, corners: list[list[float]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, "RGBA").save(OUT / f"{name}.png", optimize=True)
    (OUT / f"{name}.json").write_text(json.dumps({"coordinates": corners}, indent=2))
    print(f"{name}: {rgba.shape[1]} x {rgba.shape[0]} px")


MODELS = ["XGBoost", "CatBoost", "RandomForest", "DNN", "ANN"]
# Keep these colours in sync with the legends in app/views/maps.py
CLASSES = {0: "#E1F5EE", 1: "#9FE1CB", 2: "#FAC775", 3: "#F0997B", 4: "#D85A30"}  # Very Low .. Very High
BINARY = {0: "#D3D1C780", 1: "#185FA5"}  # non-flood half transparent, flood
ERRORS = {0: "#D3D1C780", 1: "#E24B4A", 2: "#EF9F27", 3: "#1D9E75"}  # TN, FN, FP, TP
AGREEMENT = {0: "#F1EFE880", 1: "#CECBF6", 2: "#AFA9EC", 3: "#7F77DD", 4: "#534AB7", 5: "#26215C"}

if __name__ == "__main__":
    maps = RESULTS / "maps"
    label, _ = reproject_to_wgs84(ROOT / "data" / "inventory" / "label_30m_v2.tif")
    for model in MODELS:
        values, corners = reproject_to_wgs84(maps / "probability" / f"Map_{model}.tif")
        write_layer(f"probability_{model}", colour_continuous(values, "Blues", 0, 1), corners)

        values, corners = reproject_to_wgs84(maps / "classified" / f"Classified_{model}_NaturalBreaks.tif")
        write_layer(f"classes_{model}", colour_classes(values, CLASSES), corners)

        binary, corners = reproject_to_wgs84(maps / "binary" / f"Binary_{model}_MaxCSI.tif")
        write_layer(f"binary_{model}", colour_classes(binary, BINARY), corners)

        # 0 TN, 1 FN, 2 FP, 3 TP; NaN (outside the district) stays NaN and becomes transparent
        write_layer(f"errors_{model}", colour_classes(2 * binary + label, ERRORS), corners)

    values, corners = reproject_to_wgs84(maps / "binary" / "Agreement_count_MaxCSI.tif")
    write_layer("agreement", colour_classes(values, AGREEMENT), corners)
