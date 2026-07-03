#!/usr/bin/env bash
cd /data/cl/u/xwang397/GDN-minimal-BlueVela
while :; do
  n=$(squeue -u xwang397 -h -o "%j" | grep -c 'fnl2-.*-k1024-' || true)
  [[ "$n" -eq 0 ]] && break
  sleep 120
done
echo "=== fnl2 K=1024 DONE ==="
grep -lE 'Traceback|CUDA error|out of memory|RuntimeError' slurm/fnl2/fnl2_*k1024*.log 2>/dev/null || echo "crashes: none"
for f in slurm/fnl2/fnl2_*k1024*.log; do
  [[ -e "$f" ]] || continue
  printf '%-30s %s\n' "$(basename $f)" "$(grep -aoE 'step [0-9]+: val/loss [0-9.]+ val/acc [0-9.]+' "$f" | tail -1)"
done
source /data/cl/u/xwang397/miniforge3/etc/profile.d/conda.sh && conda activate gdn
python scripts/plot_fnl2_curves.py 2>&1 | grep -viE 'triton|_cpu_device|roll back|Warning' | tail -20
echo "=== fnl2 capacity plot refreshed with K=1024 ==="
