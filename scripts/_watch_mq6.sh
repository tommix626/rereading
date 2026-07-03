#!/usr/bin/env bash
cd /data/cl/u/xwang397/GDN-minimal-BlueVela
while :; do
  n=$(squeue -u xwang397 -h -o "%j" | grep -c 'mq6-' || true)
  [[ "$n" -eq 0 ]] && break
  sleep 60
done
echo "=== ALL mq6 JOBS DONE ==="
echo "--- crashes ---"
grep -lE 'Traceback|CUDA error|RuntimeError|AssertionError' slurm/mq6/*.log 2>/dev/null || echo "none"
echo "--- final val/acc per cell ---"
for f in slurm/mq6/mq6_*.log; do
  [[ -e "$f" ]] || continue
  last=$(grep -aoE 'step [0-9]+: val/loss [0-9.]+ val/acc [0-9.]+' "$f" | tail -1)
  printf '%-34s %s\n' "$(basename "$f")" "${last:-NO_VAL_LINE}"
done
