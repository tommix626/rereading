#!/usr/bin/env bash
cd /data/cl/u/xwang397/GDN-minimal-BlueVela
while :; do
  n=$(squeue -u xwang397 -h -o "%j" | grep -c 'ol1-' || true)
  [[ "$n" -eq 0 ]] && break
  sleep 90
done
echo "=== ALL ol1 JOBS DONE ==="
echo "--- crashes ---"
grep -lE 'Traceback|CUDA error|RuntimeError|AssertionError' slurm/ol1/*.log 2>/dev/null || echo "none"
echo "--- final val/acc + early-stop step per cell ---"
for f in slurm/ol1/ol1_*.log; do
  [[ -e "$f" ]] || continue
  last=$(grep -aoE 'step [0-9]+: val/loss [0-9.]+ val/acc [0-9.]+' "$f" | tail -1)
  es=$(grep -aoE 'early-stop at step [0-9]+' "$f" | tail -1)
  printf '%-32s %s  %s\n' "$(basename "$f")" "${last:-NO_VAL}" "${es:+[$es]}"
done
