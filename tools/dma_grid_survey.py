"""Grid survey of encounter density over the DMA coverage (full-rate Danish AIS daily CSVs).

Class A vessels, last report per vessel and minute, moving = 3 <= SOG < 50 kn. For every grid cell
(0.2 deg lon x 0.1 deg lat) and day: moving vessel-minutes, distinct moving vessels, and the number of
(receiver, neighbour) pairs with separation < 1 nm / < 500 m, attributed to the receiver's cell.
Usage: python tools/dma_grid_survey.py --days 2024-08-05 ... --out results/dma_grid_survey.parquet
"""
from __future__ import annotations

import argparse
import os
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ZIPS = os.environ.get("DMA_ZIPS", "data/dma_zips")
R_E = 6371008.8
LAT0 = 56.0
DLON, DLAT = 0.2, 0.1


def one_day(day: str) -> pd.DataFrame:
    y, m, d = day.split("-")
    with zipfile.ZipFile(os.path.join(ZIPS, f"aisdk-{y}-{m}-{d}.zip")) as z:
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f, usecols=["# Timestamp", "Type of mobile", "MMSI", "Latitude", "Longitude", "SOG"],
                             engine="pyarrow")
    df = df[(df["Type of mobile"] == "Class A") & (df.SOG >= 3) & (df.SOG < 50)
            & df.Latitude.between(53, 59) & df.Longitude.between(3, 17)]
    t = pd.to_datetime(df["# Timestamp"], format="%d/%m/%Y %H:%M:%S")
    df = df.assign(minute=t.to_numpy().astype("datetime64[m]").astype("int64"))
    df = df.drop_duplicates(["MMSI", "minute"], keep="last")
    x = np.deg2rad(df.Longitude.to_numpy()) * R_E * np.cos(np.deg2rad(LAT0))
    yv = np.deg2rad(df.Latitude.to_numpy()) * R_E
    ci = np.floor(df.Longitude.to_numpy() / DLON).astype(int)
    cj = np.floor(df.Latitude.to_numpy() / DLAT).astype(int)
    cell = ci * 10000 + cj
    p1 = np.zeros(len(df), np.int64); p5 = np.zeros(len(df), np.int64)
    mins = df.minute.to_numpy()
    order = np.argsort(mins, kind="stable")
    for g in np.split(order, np.flatnonzero(np.diff(mins[order])) + 1):
        if len(g) < 2:
            continue
        pr = cKDTree(np.c_[x[g], yv[g]]).query_pairs(1852.0, output_type="ndarray")
        if len(pr) == 0:
            continue
        a, b = g[pr[:, 0]], g[pr[:, 1]]
        close = np.hypot(x[a] - x[b], yv[a] - yv[b]) < 500
        for r in (a, b):                                                       # both receivers
            np.add.at(p1, r, 1)
            np.add.at(p5, r[close], 1)
    out = pd.DataFrame(dict(cell=cell, MMSI=df.MMSI.to_numpy(), p1=p1, p5=p5))
    agg = out.groupby("cell").agg(vmin=("MMSI", "size"), vessels=("MMSI", "nunique"), p1=("p1", "sum"),
                                  p5=("p5", "sum")).reset_index()
    agg["day"] = day
    return agg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with ProcessPoolExecutor(len(a.days)) as ex:
        res = pd.concat(list(ex.map(one_day, a.days)), ignore_index=True)
    res.to_parquet(a.out)
    g = res.groupby("cell").agg(vmin=("vmin", "sum"), vessels=("vessels", "mean"), p1=("p1", "sum"),
                                p5=("p5", "sum")).reset_index()
    nd = len(a.days)
    g["lon"] = (g.cell // 10000) * DLON; g["lat"] = (g.cell % 10000) * DLAT
    g["vmin_day"] = g.vmin / nd; g["p1_rate"] = g.p1 / g.vmin; g["p5_rate"] = g.p5 / g.vmin
    g = g[g.vmin_day >= 300]
    print(g.sort_values("p1_rate", ascending=False).head(40)[["lon", "lat", "vessels", "vmin_day", "p1_rate", "p5_rate"]]
          .to_string(index=False, float_format="%.3f"))
