set -e
export HOME=/data/cl/u/xwang397
export TRITON_CACHE_DIR=/data/cl/u/xwang397/.triton
export TORCHINDUCTOR_CACHE_DIR=/data/cl/u/xwang397/.inductor
export HF_HOME=/data/cl/u/xwang397/.cache/huggingface
source /data/cl/u/xwang397/miniforge3/etc/profile.d/conda.sh
conda activate gdn
cd /data/cl/u/xwang397/GDN-minimal-BlueVela
for CFG in config/mq5_transformer.py config/mq5_gdn_noconv.py; do
  echo "########## MQ6 SMOKE $CFG ##########"
  python -u train.py $CFG \
    --dataset=mqar_zoo6_d4_n64 --block_size=64 --mqar_masked=True --vocab_size=8192 \
    --learning_rate=1e-2 --batch_size=512 --max_iters=300 \
    --decay_lr=True --warmup_iters=0 --constant_iters=0 --lr_decay_iters=300 --lr_decay_factor=0.0 \
    --eval_interval=100 --eval_iters=20 --save_checkpoint=False \
    --compile=False --out_dir=out/_mq6_smoke 2>&1 \
    | grep -aE 'step [0-9]+: val|vocab_size|Error|Traceback|assert' | tail -8
done
echo "########## MQ6 SMOKE DONE ##########"
