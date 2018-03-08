#!/bin/bash
#
# This script performs the following operations:
# 1. Fine-tunes a MobileNet model on the given training set.
# 2. Evaluates the model on the validation set.
#
# Usage:
# cd slim
# ./slim/scripts/finetune_resnet_v1_50_on_flowers.sh
set -e

MODEL_NAME=resnet_v2_50

# Where the pre-trained ResNetV1-50 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=/home/zzhang52/Libs/tensorflow/checkpoints/${MODEL_NAME}

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=/home/zzhang52/Insight/CXR/logs

# Where the dataset is saved to.
DATASET_DIR=/home/zzhang52/Insight/CXR/test

PARTITION_ID=1

PARTITION_NUM=2

NUM_EPOCHS=2

PATIENCE=2

python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_split_name=train \
  --partition_id=${PARTITION_ID} \
  --partition_num=${PARTITION_NUM} \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL_NAME} \
  --preprocessing_name=inception \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/${MODEL_NAME}.ckpt \
  --checkpoint_exclude_scopes=${MODEL_NAME}/logits \
  --trainable_scopes=${MODEL_NAME}/logits \
  --num_epochs=1 \
  --batch_size=32 \
  --learning_rate=0.01 \
  --save_interval_secs=600 \
  --save_summaries_secs=60 \
  --log_every_n_steps=100 \
  --optimizer=rmsprop \
  --weight_decay=0.00004


for epoch in `seq 1 $NUM_EPOCHS`
do
# Fine-tune only the new layers.
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_split_name=train \
  --partition_id=${PARTITION_ID} \
  --partition_num=${PARTITION_NUM} \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL_NAME} \
  --preprocessing_name=inception \
  --checkpoint_path=${TRAIN_DIR} \
  --checkpoint_exclude_scopes=${MODEL_NAME}/logits \
  --trainable_scopes=${MODEL_NAME}/logits \
  --num_epochs=1 \
  --batch_size=32 \
  --learning_rate=0.01 \
  --save_interval_secs=600 \
  --save_summaries_secs=60 \
  --log_every_n_steps=100 \
  --optimizer=rmsprop \
  --weight_decay=0.00004

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_split_name=val \
  --partition_id=${PARTITION_ID} \
  --partition_num=${PARTITION_NUM} \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL_NAME} \
  --preprocessing_name=inception \
  --eval_image_size=299 \
  --batch_size=64

# Check flag.
FLAG="$(python early_stop.py --logdir ${TRAIN_DIR} --patienc $PATIENCE)"
if [ $FLAG = stop ]
  then
  break
fi
done

for epoch in `seq 1 $NUM_EPOCHS`
do
# Fine-tune all the layers.
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR}/all \
  --dataset_split_name=train \
  --partition_id=${PARTITION_ID} \
  --partition_num=${PARTITION_NUM} \
  --dataset_dir=${DATASET_DIR} \
  --checkpoint_path=${TRAIN_DIR} \
  --model_name=${MODEL_NAME} \
  --preprocessing_name=inception \
  --num_epochs=10 \
  --batch_size=32 \
  --learning_rate=0.001 \
  --save_interval_secs=600 \
  --save_summaries_secs=60 \
  --log_every_n_steps=100 \
  --optimizer=rmsprop \
  --weight_decay=0.00004

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR}/all \
  --eval_dir=${TRAIN_DIR}/all \
  --dataset_split_name=val \
  --partition_id=${PARTITION_ID} \
  --partition_num=${PARTITION_NUM} \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL_NAME} \
  --preprocessing_name=inception \
  --eval_image_size=299 \
  --batch_size=64

# Check flag.
FLAG="$(python early_stop.py --logdir ${TRAIN_DIR}/all --patienc $PATIENCE)"
if [ $FLAG = stop ]
  then
  break
fi
done