#!/usr/bin/env python3
"""Densify the OSM Curaçao coastline with Sentinel-2 NDWI waterline crossings.

OSM rings stay the land/sea topology. Long chords are replaced by the nearest
Sentinel-2 water/land crossing along a short normal transect so the baked
waterline follows the photographed shore instead of 80–180 m OSM segments.

This is a one-shot source rebuild, not a runtime dependency.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform as warp_transform

ROOT = Path(__file__).resolve().parents[2]
OSM_PATH = ROOT / "apps/korsou/data/source/curacao-coastline-osm.json"
OUT_PATH = ROOT / "apps/korsou/data/source/curacao-coastline-rings.json"

# Curaçao explorer crop, padded so the waterline stays inside the mosaic.
UTM_BOUNDS = (477000.0, 1324000.0, 535000.0, 1376000.0)
SCENES = [
    {
        "id": "S2B_19PEP_20250220_0_L2A",
        "green": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/19/P/EP/2025/2/S2B_19PEP_20250220_0_L2A/B03.tif",
        "nir": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/19/P/EP/2025/2/S2B_19PEP_20250220_0_L2A/B08.tif",
        "scl": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/19/P/EP/2025/2/S2B_19PEP_20250220_0_L2A/SCL.tif",
    },
    {
        "id": "S2C_19PDP_20260613_0_L2A",
        "green": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/19/P/DP/2026/6/S2C_19PDP_20260613_0_L2A/B03.tif",
        "nir": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/19/P/DP/2026/6/S2C_19PDP_20260613_0_L2A/B08.tif",
        "scl": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/19/P/DP/2026/6/S2C_19PDP_20260613_0_L2A/SCL.tif",
    },
]

ALONGSHORE_M = 6.0
CORRIDOR_M = 40.0
TRANSECT_STEP_M = 2.0
MIN_POINT_SPACING_M = 4.0
CLOUD_SCL = {3, 8, 9, 10}


def assemble_osm_rings(path: Path) -> tuple[str, list[list[tuple[float, float]]]]:
    document = json.loads(path.read_text())
    nodes = {
        element["id"]: (element["lon"], element["lat"])
        for element in document["elements"]
        if element.get("type") == "node"
    }
    unused: list[list[int] | None] = []
    for element in document["elements"]:
        if element.get("type") != "way":
            continue
        if (element.get("tags") or {}).get("natural") != "coastline":
            continue
        unused.append(list(element["nodes"]))

    rings: list[list[tuple[float, float]]] = []
    while True:
        start = next((index for index, way in enumerate(unused) if way is not None), None)
        if start is None:
            break
        ring = unused[start]
        unused[start] = None
        assert ring is not None
        while ring[0] != ring[-1]:
            end = ring[-1]
            next_index = next(
                (
                    index
                    for index, way in enumerate(unused)
                    if way is not None and (way[0] == end or way[-1] == end)
                ),
                None,
            )
            if next_index is None:
                raise RuntimeError(f"OSM coastline is not a closed ring at node {end}")
            nxt = unused[next_index]
            unused[next_index] = None
            assert nxt is not None
            if nxt[-1] == end:
                nxt.reverse()
            ring.extend(nxt[1:])
        rings.append([nodes[node_id] for node_id in ring])
    return document["osm3s"]["timestamp_osm_base"], rings


def lonlat_to_utm(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    easting, northing = warp_transform("EPSG:4326", "EPSG:32619", lon, lat)
    return np.asarray(easting), np.asarray(northing)


def utm_to_lonlat(easting: np.ndarray, northing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon, lat = warp_transform("EPSG:32619", "EPSG:4326", easting, northing)
    return np.asarray(lon), np.asarray(lat)


class NdwiMosaic:
    def __init__(self, scenes: list[dict], bounds: tuple[float, float, float, float]):
        west, south, east, north = bounds
        self.origin_e = west
        self.origin_n = north
        self.step = 10.0
        self.width = int(math.ceil((east - west) / self.step))
        self.height = int(math.ceil((north - south) / self.step))
        self.green = np.full((self.height, self.width), np.nan, dtype=np.float32)
        self.nir = np.full((self.height, self.width), np.nan, dtype=np.float32)
        self.cloud = np.zeros((self.height, self.width), dtype=bool)
        self.scene_ids = [scene["id"] for scene in scenes]
        for scene in scenes:
            self._paint(scene, bounds)
        valid = np.isfinite(self.green) & np.isfinite(self.nir) & (self.green + self.nir > 0)
        self.ndwi = np.full_like(self.green, np.nan)
        self.ndwi[valid] = (self.green[valid] - self.nir[valid]) / (
            self.green[valid] + self.nir[valid]
        )

    def _paint(self, scene: dict, bounds: tuple[float, float, float, float]) -> None:
        west, south, east, north = bounds
        with rasterio.open(scene["green"]) as green_src, rasterio.open(scene["nir"]) as nir_src:
            window = from_bounds(west, south, east, north, green_src.transform)
            green = green_src.read(1, window=window, boundless=True, fill_value=0).astype(
                np.float32
            )
            nir = nir_src.read(1, window=window, boundless=True, fill_value=0).astype(np.float32)
            transform = green_src.window_transform(window)
        with rasterio.open(scene["scl"]) as scl_src:
            scl_window = from_bounds(west, south, east, north, scl_src.transform)
            scl = scl_src.read(1, window=scl_window, boundless=True, fill_value=0)
            scl_transform = scl_src.window_transform(scl_window)

        rows, cols = np.indices(green.shape)
        easting = transform.c + (cols + 0.5) * transform.a
        northing = transform.f + (rows + 0.5) * transform.e
        dst_col = np.floor((easting - self.origin_e) / self.step).astype(np.int32)
        dst_row = np.floor((self.origin_n - northing) / self.step).astype(np.int32)
        inside = (
            (dst_col >= 0)
            & (dst_col < self.width)
            & (dst_row >= 0)
            & (dst_row < self.height)
            & (green > 0)
            & (nir > 0)
        )
        self.green[dst_row[inside], dst_col[inside]] = green[inside] / 10_000.0
        self.nir[dst_row[inside], dst_col[inside]] = nir[inside] / 10_000.0

        scl_rows, scl_cols = np.indices(scl.shape)
        scl_e = scl_transform.c + (scl_cols + 0.5) * scl_transform.a
        scl_n = scl_transform.f + (scl_rows + 0.5) * scl_transform.e
        scl_dst_col = np.floor((scl_e - self.origin_e) / self.step).astype(np.int32)
        scl_dst_row = np.floor((self.origin_n - scl_n) / self.step).astype(np.int32)
        scl_inside = (
            (scl_dst_col >= 0)
            & (scl_dst_col < self.width)
            & (scl_dst_row >= 0)
            & (scl_dst_row < self.height)
        )
        cloudy = np.isin(scl, list(CLOUD_SCL))
        self.cloud[scl_dst_row[scl_inside], scl_dst_col[scl_inside]] |= cloudy[scl_inside]

    def sample(self, easting: float, northing: float) -> tuple[float, bool]:
        col = (easting - self.origin_e) / self.step
        row = (self.origin_n - northing) / self.step
        if col < 0 or row < 0 or col >= self.width - 1 or row >= self.height - 1:
            return math.nan, True
        c0 = int(col)
        r0 = int(row)
        if self.cloud[r0, c0] or self.cloud[min(r0 + 1, self.height - 1), min(c0 + 1, self.width - 1)]:
            return math.nan, True
        tc = col - c0
        tr = row - r0
        values = (
            self.ndwi[r0, c0],
            self.ndwi[r0, c0 + 1],
            self.ndwi[r0 + 1, c0],
            self.ndwi[r0 + 1, c0 + 1],
        )
        if any(not np.isfinite(value) for value in values):
            return math.nan, False
        top = values[0] * (1.0 - tc) + values[1] * tc
        bottom = values[2] * (1.0 - tc) + values[3] * tc
        return float(top * (1.0 - tr) + bottom * tr), False


def ring_utm(ring: list[tuple[float, float]]) -> np.ndarray:
    lon = np.array([point[0] for point in ring], dtype=np.float64)
    lat = np.array([point[1] for point in ring], dtype=np.float64)
    easting, northing = lonlat_to_utm(lon, lat)
    return np.column_stack([easting, northing])


def densify_ring(ring_ll: list[tuple[float, float]], mosaic: NdwiMosaic) -> list[list[float]]:
    utm = ring_utm(ring_ll)
    samples: list[np.ndarray] = [utm[0]]
    for start, end in zip(utm[:-1], utm[1:]):
        chord = end - start
        length = float(np.hypot(*chord))
        if length < 1.0e-3:
            continue
        tangent = chord / length
        # OSM coastline has land on the left, water on the right.
        waterward = np.array([tangent[1], -tangent[0]])
        steps = max(1, int(math.ceil(length / ALONGSHORE_M)))
        for step in range(1, steps + 1):
            t = step / steps
            base = start + chord * t
            crossing = transect_crossing(mosaic, base, waterward)
            samples.append(crossing if crossing is not None else base)

    simplified = [samples[0]]
    for point in samples[1:]:
        if float(np.hypot(*(point - simplified[-1]))) >= MIN_POINT_SPACING_M:
            simplified.append(point)
    if float(np.hypot(*(simplified[0] - simplified[-1]))) > 1.0e-3:
        simplified.append(simplified[0].copy())
    else:
        simplified[-1] = simplified[0].copy()

    easting = np.array([point[0] for point in simplified])
    northing = np.array([point[1] for point in simplified])
    lon, lat = utm_to_lonlat(easting, northing)
    return [[float(x), float(y)] for x, y in zip(lon, lat)]


def transect_crossing(
    mosaic: NdwiMosaic, origin: np.ndarray, waterward: np.ndarray
) -> np.ndarray | None:
    offsets = np.arange(-CORRIDOR_M, CORRIDOR_M + TRANSECT_STEP_M * 0.5, TRANSECT_STEP_M)
    values: list[tuple[float, float]] = []
    for offset in offsets:
        point = origin + waterward * offset
        ndwi, cloudy = mosaic.sample(float(point[0]), float(point[1]))
        if cloudy or not math.isfinite(ndwi):
            continue
        values.append((float(offset), ndwi))
    for (offset_a, ndwi_a), (offset_b, ndwi_b) in zip(values, values[1:]):
        if (ndwi_a >= 0.0) == (ndwi_b >= 0.0):
            continue
        span = ndwi_b - ndwi_a
        if abs(span) < 1.0e-6:
            continue
        t = ndwi_a / (ndwi_a - ndwi_b)
        offset = offset_a + (offset_b - offset_a) * t
        return origin + waterward * offset
    return None


def main() -> None:
    timestamp, rings = assemble_osm_rings(OSM_PATH)
    print(f"OSM rings {len(rings)} timestamp {timestamp}")
    mosaic = NdwiMosaic(SCENES, UTM_BOUNDS)
    valid = np.isfinite(mosaic.ndwi)
    print(
        f"NDWI mosaic {mosaic.width}×{mosaic.height}, "
        f"{valid.mean() * 100:.1f}% valid, cloud {mosaic.cloud.mean() * 100:.1f}%"
    )
    densified = [ring for ring in (densify_ring(ring, mosaic) for ring in rings) if len(ring) >= 4]
    before = sum(len(ring) for ring in rings)
    after = sum(len(ring) for ring in densified)
    print(f"vertices {before} -> {after}")
    document = {
        "format": "korsou.coastline-rings.v1",
        "crs": "EPSG:4326",
        "osm_source": OSM_PATH.name,
        "osm_timestamp": timestamp,
        "sentinel": {
            "scenes": mosaic.scene_ids,
            "bands": ["B03", "B08"],
            "collection": "sentinel-2-l2a",
            "method": "NDWI zero-crossing along OSM-normal transects",
            "alongshore_m": ALONGSHORE_M,
            "corridor_m": CORRIDOR_M,
            "source_url": "https://registry.opendata.aws/sentinel-2-l2a-cogs/",
            "attribution": "Contains modified Copernicus Sentinel-2 data (ESA), processed to L2A COGs by Element 84 / AWS Earth Search",
        },
        "rings": densified,
    }
    OUT_PATH.write_text(json.dumps(document))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
