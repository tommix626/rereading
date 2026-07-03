set -e
export HOME=/data/cl/u/xwang397
export TRITON_CACHE_DIR=/data/cl/u/xwang397/.triton
export TORCHINDUCTOR_CACHE_DIR=/data/cl/u/xwang397/.inductor
export HF_HOME=/data/cl/u/xwang397/.cache/huggingface
source /data/cl/u/xwang397/miniforge3/etc/profile.d/conda.sh
conda activate gdn
cd /data/cl/u/xwang397/GDN-minimal-BlueVela
for CFG in config/mq5_transformer.py config/mq5_gdn_noconv.py; do
  echo "########## OL1 SMOKE $CFG ##########"
  python -u train.py $CFG \
    --dataset=online_mqar --block_size=64 --mqar_masked=True --vocab_size=8192 \
    --online_mqar=True --online_num_kv_pairs=4 --online_power_a=0.01 --online_random_non_queries=False \
    --learning_rate=1e-2 --batch_size=512 --max_iters=1500 --decay_lr=False \
    --early_stop_acc=0.30 --early_stop_window=2 \
    --eval_interval=100 --eval_iters=20 --save_checkpoint=False \
    --compile=False --out_dir=out/_ol1_smoke 2>&1 \
    | grep -aE 'online\]|step [0-9]+: val|early-stop|Error|Traceback|assert' | tail -14
done
echo "########## OL1 SMOKE DONE ##########"
