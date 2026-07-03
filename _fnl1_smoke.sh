set -e
export HOME=/data/cl/u/xwang397
export TRITON_CACHE_DIR=/data/cl/u/xwang397/.triton
export TORCHINDUCTOR_CACHE_DIR=/data/cl/u/xwang397/.inductor
export HF_HOME=/data/cl/u/xwang397/.cache/huggingface
source /data/cl/u/xwang397/miniforge3/etc/profile.d/conda.sh
conda activate gdn
cd /data/cl/u/xwang397/GDN-minimal-BlueVela
echo "##### SOFTMAX (RoPE) #####"
python -u train.py config/mq5_transformer.py --use_rope=True --use_learned_pos_emb=False \
  --dataset=mqar_fnl1_k16_q16 --block_size=64 --mqar_masked=True --vocab_size=8192 \
  --learning_rate=1e-2 --batch_size=256 --max_iters=400 \
  --eval_interval=100 --eval_iters=20 --save_checkpoint=False --compile=False --out_dir=out/_fnl1_smoke_sm 2>&1 \
  | grep -aE 'Overriding: use_rope|Overriding: use_learned|step [0-9]+: val|Error|Traceback' | tail -8
echo "##### GDN (noconv) #####"
python -u train.py config/mq5_gdn_noconv.py \
  --dataset=mqar_fnl1_k16_q16 --block_size=64 --mqar_masked=True --vocab_size=8192 \
  --learning_rate=1e-2 --batch_size=256 --max_iters=400 \
  --eval_interval=100 --eval_iters=20 --save_checkpoint=False --compile=False --out_dir=out/_fnl1_smoke_gdn 2>&1 \
  | grep -aE 'step [0-9]+: val|Error|Traceback' | tail -6
echo "##### FNL1 SMOKE DONE #####"
