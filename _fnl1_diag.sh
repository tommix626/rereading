set -e
export HOME=/data/cl/u/xwang397
export TRITON_CACHE_DIR=/data/cl/u/xwang397/.triton
export TORCHINDUCTOR_CACHE_DIR=/data/cl/u/xwang397/.inductor
export HF_HOME=/data/cl/u/xwang397/.cache/huggingface
source /data/cl/u/xwang397/miniforge3/etc/profile.d/conda.sh
conda activate gdn
cd /data/cl/u/xwang397/GDN-minimal-BlueVela
LR=1e-3
run() {  # name dataset block
  local name=$1 ds=$2 blk=$3
  echo "########## $name (softmax RoPE, lr=$LR, wd=0.1, 2L d=64, 25k) ##########"
  python -u train.py config/mq5_transformer.py --use_rope=True --use_learned_pos_emb=False \
    --dataset=$ds --block_size=$blk --mqar_masked=True --vocab_size=8192 \
    --learning_rate=$LR --batch_size=256 --max_iters=25000 \
    --eval_interval=1000 --eval_iters=50 --save_checkpoint=False --compile=False \
    --out_dir=out/_diag_$name 2>&1 \
    | grep -aoE 'step [0-9]+: val/loss [0-9.]+ val/acc [0-9.]+' | awk 'NR==1||NR%4==0||NR>22'
}
run fnl1_k2  mqar_fnl1_k2_q2   8
run fnl1_k4  mqar_fnl1_k4_q4   16
run fnl1_k8  mqar_fnl1_k8_q8   32
run CONTROL_zoo_d32 mqar_zoo_d32_n192 192   # report setup: expect ~0.999
echo "########## DIAG DONE ##########"
