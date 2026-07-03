#!/usr/bin/env bash
cd /data/cl/u/xwang397/GDN-minimal-BlueVela
while :; do
  n=$(squeue -u xwang397 -h -o "%j" | grep -c 'fnl1-' || true)
  [[ "$n" -eq 0 ]] && break
  sleep 90
done
echo "=== ALL fnl1 JOBS DONE ==="
grep -lE 'Traceback|CUDA error|RuntimeError|AssertionError' slurm/fnl1/*.log 2>/dev/null || echo "crashes: none"
for f in slurm/fnl1/fnl1_*.log; do
  [[ -e "$f" ]] || continue
  printf '%-30s %s\n' "$(basename "$f")" "$(grep -aoE 'step [0-9]+: val/loss [0-9.]+ val/acc [0-9.]+' "$f" | tail -1)"
done
