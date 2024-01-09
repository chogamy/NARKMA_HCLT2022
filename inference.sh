CUDA_VISIBLE_DEVICES=1 python -W ignore inference.py \
  --hparams logs/tb_logs/default/version_0/hparams.yaml \
  --model_binary logs/last.ckpt \
  --testfile data/test --outputfile infer.txt --length_beam_size 5 