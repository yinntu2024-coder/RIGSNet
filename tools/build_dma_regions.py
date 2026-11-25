"""Extract the five dense-traffic regions from the full-rate DMA daily archives (2024-08-01 .. 2024-10-31)
and run the shared pre-processing protocol (aistraj/data/preprocess.py) on each.

Stage 1 (extract): Class A position reports inside each region box -> data/raw_dma/<R>/<day>.parquet
Stage 2 (raw):     concat -> data/raw_dma/<R>.csv (mmsi, t, lon, lat, sog, cog, length) and
                   data/raw_dma/<R>_dims.csv (median reported length / width per MMSI)
Stage 3 (process): aistraj.data.preprocess.preprocess_csv -> data/processed/<R>/tracks.npz, meta.json
Stage 4 (dims):    data/processed/<R>/seg_dims.npz: vessel length / beam per segment (rigs.dims)
Region boxes were chosen from the encounter-density grid survey (tools/dma_grid_survey.py).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


ZIPS = os.environ.get("DMA_ZIPS", "data/dma_zips")
RAW = os.path.join(ROOT, "data", "raw_dma")
PROC = os.path.join(ROOT, "data", "processed")
DAYS = [(date(2024, 8, 1) + timedelta(d)).isoformat() for d in range(92)]

# lon_min, lon_max, lat_min, lat_max
REGIONS = {
    "RWA": (11.8, 12.4, 54.0, 54.3),    # Rostock-Warnemuende
    "ESB": (8.2, 8.6, 55.3, 55.6),      # Esbjerg
    "KIB": (9.8, 10.4, 54.2, 54.5),     # Kiel Bight
    "GOT": (11.4, 12.0, 57.5, 57.8),    # Gothenburg approach
    "ORN": (12.4, 12.8, 55.9, 56.2),    # Oresund North (Helsingor-Helsingborg)
}
COLS = {"# Timestamp": "ts", "Type of mobile": "mobile", "MMSI": "mmsi", "Latitude": "lat", "Longitude": "lon",
        "SOG": "sog", "COG": "cog", "Width": "width", "Length": "length"}


def extract_day(day: str) -> dict:
    todo = {r: os.path.join(RAW, r, f"{day}.parquet") for r in REGIONS}
    if all(os.path.exists(p) for p in todo.values()):
        return {}
    y, m, d = day.split("-")
    with zipfile.ZipFile(os.path.join(ZIPS, f"aisdk-{y}-{m}-{d}.zip")) as z:
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f, usecols=list(COLS), engine="pyarrow").rename(columns=COLS)
    df = df[df.mobile == "Class A"].drop(columns="mobile")
    df["t"] = pd.to_datetime(df.ts, format="%d/%m/%Y %H:%M:%S", utc=True).astype("int64") // 10**9
    df = df.drop(columns="ts")
    n = {}
    for r, (a, b, c, e) in REGIONS.items():
        s = df[df.lon.between(a, b) & df.lat.between(c, e)]
        os.makedirs(os.path.dirname(todo[r]), exist_ok=True)
        s.to_parquet(todo[r] + ".tmp", index=False)
        os.replace(todo[r] + ".tmp", todo[r])
        n[r] = len(s)
    return n


def build_raw(r: str):
    df = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(os.path.join(RAW, r, "*.parquet")))],
                   ignore_index=True)
    dims = df.groupby("mmsi")[["length", "width"]].median()
    dims.to_csv(os.path.join(RAW, f"{r}_dims.csv"))
    df[["mmsi", "t", "lon", "lat", "sog", "cog", "length"]].to_csv(os.path.join(RAW, f"{r}.csv"), index=False)
    return r, len(df), int(df.mmsi.nunique())


def process(r: str):
    from aistraj.data.preprocess import preprocess_csv
    log = preprocess_csv(os.path.join(RAW, f"{r}.csv"), os.path.join(PROC, r))
    return r, log


def dims(r: str):
    from rigs.config import RConfig
    from rigs.dims import clean_dims
    cfg = RConfig()
    seg_mmsi = np.load(os.path.join(PROC, r, "tracks.npz"))["seg_mmsi"]
    tab = pd.read_csv(os.path.join(RAW, f"{r}_dims.csv"), index_col="mmsi")
    L, B, ok = clean_dims(tab.length.reindex(seg_mmsi).to_numpy(), tab.width.reindex(seg_mmsi).to_numpy(),
                          cfg.L_default, cfg.B_default)
    np.savez(os.path.join(PROC, r, "seg_dims.npz"), L=L.astype(np.float32), B=B.astype(np.float32), known=ok)
    return r, dict(segments=len(ok), known=float(ok.mean()), L_median=float(np.median(L)), B_median=float(np.median(B)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["extract", "raw", "process", "dims"], required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--regions", nargs="*", default=list(REGIONS))
    a = ap.parse_args()
    if a.stage == "extract":
        with ProcessPoolExecutor(a.workers) as ex:
            for day, n in zip(DAYS, ex.map(extract_day, DAYS)):
                print(day, n, flush=True)
    elif a.stage == "raw":
        with ProcessPoolExecutor(len(a.regions)) as ex:
            for res in ex.map(build_raw, a.regions):
                print(*res, flush=True)
    elif a.stage == "process":
        with ProcessPoolExecutor(len(a.regions)) as ex:
            for r, log in ex.map(process, a.regions):
                print(r, json.dumps(log), flush=True)
    else:
        for r in a.regions:
            print(*dims(r), flush=True)
