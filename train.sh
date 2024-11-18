CUDA_VISIBLE_DEVICES=0 python trainer.py \
--ckpt_path "./ckpts" \
--log_dir "./runs" \
--batch_size 16 \
--lr 0.0001 \
--train_path "path/to/dataset" \
--eval_path "path/to/eval/dataset" \