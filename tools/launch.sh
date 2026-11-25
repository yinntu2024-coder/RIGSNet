#!/bin/bash
# usage: tools/launch.sh <runs_dir> <eval_split> <gpus "0 1 2 3"> <job files...>
cd "$(dirname "$0")/.." && mkdir -p logs
RUNS=$1; SPLIT=$2; GPUS=$3; shift 3
PY=${PY:-python}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 2026-11-21: cut fragmentation OOM on dense GOT/ORN (allocator only)
[ "$CLEAN" = 1 ] && for c in $(find "$RUNS" -name .claim -type d 2>/dev/null); do d=$(dirname "$c"); [ -f "$d/${SPLIT}_metrics.json" ] || rm -rf "$d"; done
for g in $GPUS; do
  CUDA_VISIBLE_DEVICES=$g nohup $PY tools/run_jobs.py --jobs "$@" --runs "$RUNS" --eval-split "$SPLIT" >> logs/$(basename "$RUNS")_gpu$g.log 2>&1 &
  echo $! >> logs/workers.pid
done
