set -e
export HOME=/data/cl/u/xwang397
export TRITON_CACHE_DIR=/data/cl/u/xwang397/.triton
export TORCHINDUCTOR_CACHE_DIR=/data/cl/u/xwang397/.inductor
export HF_HOME=/data/cl/u/xwang397/.cache/huggingface
source /data/cl/u/xwang397/miniforge3/etc/profile.d/conda.sh
conda activate gdn
cd /data/cl/u/xwang397/GDN-minimal-BlueVela
run() {  # name dataset block lr
  echo "########## $1 (softmax RoPE, lr=$4, wd=0.1, 2L d=64, 25k) ##########"
  python -u train.py config/mq5_transformer.py --use_rope=True --use_learned_pos_emb=False \
    --dataset=$2 --block_size=$3 --mqar_masked=True --vocab_size=8192 \
    --learning_rate=$4 --batch_size=256 --max_iters=25000 \
    --eval_interval=1000 --eval_iters=50 --save_checkpoint=False --compile=False \
    --out_dir=out/_fnl2t_$1 2>&1 | grep -aoE 'step [0-9]+: val/loss [0-9.]+ val/acc [0-9.]+' | awk 'NR==1||NR%5==0||NR>23'
}
# fnl2 = compact K/Q + RANDOM fillers. Expect softmax to now SOLVE (vs fnl1 1/K plateau).
run fnl2_k8_lr1e-3  mqar_fnl2_k8_q8   32 1e-3
run fnl2_k32_lr1e-3 mqar_fnl2_k32_q32 128 1e-3
run fnl2_k32_lr2e-3 mqar_fnl2_k32_q32 128 2.1544e-3
echo "########## FNL2 TEST DONE ##########"
