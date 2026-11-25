#!/usr/bin/env python3
"""Worker that walks a JSON job list and claims jobs with an atomic mkdir, so several workers (one per
GPU) can share one list and be restarted at any time.

    CUDA_VISIBLE_DEVICES=0 python tools/run_jobs.py --jobs jobs/main.json --runs runs &
Each job: {"dataset": .., "model": .., "variant": "full", "tag": "", "seed": 0, "overrides": [..]}.
"""
import argparse
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch  # noqa: E402

from rigs.experiment import run_dir, run_experiment  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", nargs="+", required=True)
    ap.add_argument("--runs", default="runs_dma")
    ap.add_argument("--eval-split", default="test", choices=["test", "val"])
    # 2026-11-21: markers this worker may take over (HOLD_got = GOT jobs reserved for tools/got_sched.sh, one GOT per GPU)
    ap.add_argument("--take-markers", default="HOLD_old_code,HOLD_requeue")
    a = ap.parse_args()
    todo = [j for f in a.jobs for j in json.load(open(f))]
    for _ in range(2):
        for j in todo:
            out = run_dir(a.runs, j["dataset"], j["model"], j.get("variant", "full"), j.get("tag", ""), j.get("seed", 0))
            if os.path.exists(os.path.join(out, f"{a.eval_split}_metrics.json")):
                continue
            os.makedirs(out, exist_ok=True)
            try:
                os.mkdir(os.path.join(out, ".claim"))
            except FileExistsError:
                # placeholders that keep workers with stale code away; first new worker takes it.
                # HOLD_requeue (2026-11-21): unknown to workers started before it existed, so those drain and exit.
                for m in a.take_markers.split(","):
                    try:
                        os.remove(os.path.join(out, m))
                        break
                    except FileNotFoundError:
                        pass
                else:
                    continue
            try:
                run_experiment(j["dataset"], j["model"], j.get("variant", "full"), j.get("tag", ""), j.get("seed", 0),
                               a.runs, j.get("overrides", []), a.eval_split,
                               log=lambda s: print(s, flush=True))
            except Exception:
                traceback.print_exc()
                print(f"[fail] {out}", flush=True)
                try:
                    os.rmdir(os.path.join(out, ".claim"))
                except OSError:
                    pass
                torch.cuda.empty_cache()
