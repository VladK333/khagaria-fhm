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


def write_layer(name: str, rgba: np.ndarray, corners: list[list[float]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, "RGBA").save(OUT / f"{name}.png", optimize=True)
    (OUT / f"{name}.json").write_text(json.dumps({"coordinates": corners}, indent=2))
    print(f"{name}: {rgba.shape[1]} x {rgba.shape[0]} px -> {OUT / name}.png")


if __name__ == "__main__":
    values, corners = reproject_to_wgs84(RESULTS / "maps" / "probability" / "Map_XGBoost.tif")
    write_layer("probability_XGBoost", colour_continuous(values, "Blues", 0, 1), corners)
