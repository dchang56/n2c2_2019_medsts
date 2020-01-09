#!/bin/bash

i=0
for EPOCHS in 4 5 6 7 8; do
	for k in 0 1 2 4; do

	OUTPUT_DIR=/home/dc925/project/medsts/output/full_nepochs/nepochs_${i}/k_fold_${k}

	mkdir -p $OUTPUT_DIR

	python /home/dc925/project/medsts/code/main.py \
	--text_encoder_checkpoint /home/dc925/project/medsts/output/finetuning \
	--graph_encoder_checkpoint /home/dc925/project/medsts/notebooks/saved_model_68.pth \
	--config_name /home/dc925/project/medsts/output/finetuning/config.json \
	--do_train \
	--do_eval \
	--task_name medsts \
	--data_dir /home/dc925/project/data/seq_pair/MEDSTS/k_fold_${k} \
	--output_dir=$OUTPUT_DIR \
	--per_gpu_train_batch_size=12 \
	--learning_rate=1.8e-4 \
	--weight_decay 5e-4 \
	--num_train_epochs=$EPOCHS \
	--overwrite_cache \
	--overwrite_output_dir \
	--evaluate_during_training \
	--do_lower_case \
	--df \
	--augment

	echo epochs $EPOCHS > $OUTPUT_DIR/hyperparams.txt

	done
i=$((i+1))
done

