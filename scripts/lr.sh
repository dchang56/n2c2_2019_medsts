#!/bin/bash

i=0
for LR in 5e-5 9e-5 1.3e-4 1.7e-4 2e-4 2.4e-4 2.8e-4; do
	for k in 0 1 2 3 4; do

	OUTPUT_DIR=/home/dc925/project/medsts/output/checkpoint_lr_test/lr_run_${i}/k_fold_${k}

	mkdir -p $OUTPUT_DIR

	python /home/dc925/project/medsts/code/main.py \
	--text_encoder_checkpoint /home/dc925/project/medsts/output/finetuned_${k} \
	--graph_encoder_checkpoint /home/dc925/project/data/seq_pair/MEDSTS/k_fold_${k}/graph_encoder_checkpoint.pth \
	--config_name /home/dc925/project/medsts/output/finetuned_${k}/config.json \
	--do_train \
	--do_eval \
	--task_name medsts \
	--data_dir /home/dc925/project/data/seq_pair/MEDSTS/k_fold_${k} \
	--output_dir=$OUTPUT_DIR \
	--per_gpu_train_batch_size=12 \
	--learning_rate=$LR \
	--graph_lr 1e-3 \
	--weight_decay 5e-4 \
	--num_train_epochs 5 \
	--overwrite_cache \
	--overwrite_output_dir \
	--evaluate_during_training \
	--do_lower_case \
	--df \
	--augment \
	--warmup_steps 200

	echo lr $LR > $OUTPUT_DIR/hyperparams.txt

	done
i=$((i+1))
done


