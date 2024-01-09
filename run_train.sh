CUDA_VISIBLE_DEVICES=0,1 python train_nat.py \
  --max_epochs 400 --warmup_ratio 0.05 \
  --batch_size 128 --max_len 200 \
  --num_workers 8 --lr 5e-4 \
  --default_root_dir logs  --gpus 2 \
  --train_file data/train --valid_file data/valid 