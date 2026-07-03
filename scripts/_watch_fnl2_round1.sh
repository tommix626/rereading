#!/usr/bin/env bash
cd /data/cl/u/xwang397/GDN-minimal-BlueVela
while :; do
  n=$(squeue -u xwang397 -h -o "%j" | grep -cE 'fnl2-(softmax|gdn)-k(16|32|64|128)-' || true)
  [[ "$n" -eq 0 ]] && break
  sleep 90
done
echo "=== fnl2 K<=128 DONE (FIRST ROUND) ==="
source /data/cl/u/xwang397/miniforge3/etc/profile.d/conda.sh && conda activate gdn
FNL2_KS="16 32 64 128" python scripts/plot_fnl2_curves.py 2>&1 | grep -viE 'triton|_cpu_device|roll back|Warning' | tail -25
cp -f slurm/plots/fnl2_acc_vs_K.png slurm/plots/fnl2_acc_vs_K_round1.png 2>/dev/null || true
cp -f slurm/plots/fnl2_all_lr_curves.png slurm/plots/fnl2_all_lr_curves_round1.png 2>/dev/null || true
echo "=== round1 fnl2 plots saved ==="
