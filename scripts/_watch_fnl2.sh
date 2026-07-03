#!/usr/bin/env bash
cd /data/cl/u/xwang397/GDN-minimal-BlueVela
while :; do
  n=$(squeue -u xwang397 -h -o "%j" | grep -c 'fnl2-' || true)
  [[ "$n" -eq 0 ]] && break
  sleep 90
done
echo "=== ALL fnl2 JOBS DONE ==="
grep -lE 'Traceback|CUDA error|RuntimeError|AssertionError' slurm/fnl2/*.log 2>/dev/null || echo "crashes: none"
source /data/cl/u/xwang397/miniforge3/etc/profile.d/conda.sh && conda activate gdn
python scripts/plot_fnl2_curves.py 2>&1 | grep -viE 'triton|_cpu_device|roll back|Warning' | tail -30
echo "=== fnl2 plots written to slurm/plots/fnl2_*.png ==="
