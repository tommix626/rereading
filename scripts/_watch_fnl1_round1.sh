#!/usr/bin/env bash
# First-round: fire when all fnl1 K<=128 jobs are done; plot just those K.
cd /data/cl/u/xwang397/GDN-minimal-BlueVela
while :; do
  n=$(squeue -u xwang397 -h -o "%j" | grep -cE 'fnl1-(softmax|gdn)-k(16|32|64|128)-' || true)
  [[ "$n" -eq 0 ]] && break
  sleep 90
done
echo "=== fnl1 K<=128 ALL DONE (FIRST ROUND) ==="
source /data/cl/u/xwang397/miniforge3/etc/profile.d/conda.sh && conda activate gdn
FNL1_KS="16 32 64 128" python scripts/plot_fnl1_curves.py 2>&1 | grep -viE 'triton|_cpu_device|roll back|Warning' | tail -30
# preserve round-1 copies (final run will overwrite the base names)
cp -f slurm/plots/fnl1_acc_vs_K.png       slurm/plots/fnl1_acc_vs_K_round1.png 2>/dev/null || true
cp -f slurm/plots/fnl1_all_lr_curves.png  slurm/plots/fnl1_all_lr_curves_round1.png 2>/dev/null || true
echo "=== round1 plots saved: slurm/plots/fnl1_*_round1.png ==="
