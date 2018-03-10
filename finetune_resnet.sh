#!/bin/bash
#
# This script performs the following operations:
# 1. Fine-tunes a MobileNet model on the given training set.
# 2. Evaluates the model on the validation set.
#
# Usage:
# cd slim
# ./slim/scripts/finetune_resnet_v1_50_on_flowers.sh

#-----------------SLURM Parameters-------------------------------
#SBATCH -p terramepp
#SBATCH -n 8
#SBATCH --mem=64g
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -D /home/n-z/zzhang52/Insight/ChestXray/
#SBATCH -J resnet_50_small_train
#SBATCH -A ChestXray
#PBS -M zzhang52@illinois.edu
#PBS -m abe

set -e

#-----------------Load Modules------------------------------------
module load Tensorflow-GPU/1.2.1-IGB-gcc-4.9.4-Python-3.6.1
export PYTHONPATH="$PYTHONPATH:/home/n-z/tensorflow/models/research/slim"

#-----------------Define Variables-------------------------------
MODEL_NAME=resnet_v2_50

# Where the pre-trained ResNetV1-50 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=/home/n-z/zzhang52/tensorflow/checkpoints/${MODEL_NAME}

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=/home/n-z/zzhang52/Insight/CXR/temp

# Where the dataset is saved to.
DATASET_DIR=/home/n-z/zzhang52/Insight/CXR/small

PARTITION_ID=1

PARTITION_NUM=1

NUM_EPOCHS=400

NUM_EPOSCH_ALL=200

PATIENCE=100

BATCH_SIZE=64

EVAL_EVERY_EPOCH=3

NUM_EPOCHS=$(( NUM_EPOCHS / EVAL_EVERY_EPOCH ))
NUM_EPOCHS_ALL=$(( NUM_EPOCHS_ALL / EVAL_EVERY_EPOCH ))

IMAGE_SIZE=299

#-----------------------Execute Commands------------------------------

python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_split_name=train \
  --partition_id=${PARTITION_ID} \
  --partition_num=${PARTITION_NUM} \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL_NAME} \
  --preprocessing_name=inception \
  --train_image_size=$IMAGE_SIZE} \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/${MODEL_NAME}.ckpt \
  --checkpoint_exclude_scopes=${MODEL_NAME}/logits \
  --trainable_scopes=${MODEL_NAME}/logits \
  --num_epochs=1 \
  --batch_size=${BATCH_SIZE} \
  --learning_rate=0.001 \
  --save_interval_secs=0 \
  --save_summaries_secs=60 \
  --log_every_n_steps=100 \
  --optimizer=adam \
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
  --train_image_size=${IMAGE_SIZE} \
  --checkpoint_path=${TRAIN_DIR} \
  --checkpoint_exclude_scopes=${MODEL_NAME}/logits \
  --trainable_scopes=${MODEL_NAME}/logits \
  --num_epochs=${EVAL_EVERY_EPOCH} \
  --batch_size=${BATCH_SIZE} \
  --learning_rate=0.001 \
  --save_interval_secs=0 \
  --save_summaries_secs=300 \
  --log_every_n_steps=100 \
  --optimizer=adam \
  --weight_decay=0.00004 \
  -learning_rate_decay_factor=0.9 \

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_split_name=train \
  --partition_id=${PARTITION_ID} \
  --partition_num=${PARTITION_NUM} \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL_NAME} \
  --preprocessing_name=inception \
  --eval_image_size=${IMAGE_SIZE} \
  --batch_size=${BATCH_SIZE}

python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_split_name=val \
  --partition_id=${PARTITION_ID} \
  --partition_num=${PARTITION_NUM} \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL_NAME} \
  --preprocessing_name=inception \
  --eval_image_size=${IMAGE_SIZE} \
  --batch_size=${BATCH_SIZE}

# Check flag.
FLAG="$(python early_stop.py --logdir ${TRAIN_DIR} --patience $PATIENCE)"
if [ $FLAG = stop ]
  then
  break
fi
done

for epoch in `seq 1 $NUM_EPOCHS_ALL`
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
  --train_image_size=${IMAGE_SIZE} \
  --num_epochs=${EVAL_EVERY_EPOCH} \
  --batch_size=${BATCH_SIZE} \
  --learning_rate=0.001 \
  --save_interval_secs=0 \
  --save_summaries_secs=300 \
  --log_every_n_steps=100 \
  --optimizer=adam \
  --weight_decay=0.00004 \
  --learning_rate_decay_factor=0.9

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR}/all \
  --eval_dir=${TRAIN_DIR}/all \
  --dataset_split_name=train \
  --partition_id=${PARTITION_ID} \
  --partition_num=${PARTITION_NUM} \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL_NAME} \
  --preprocessing_name=inception \
  --eval_image_size=${IMAGE_SIZE} \
  --batch_size=${BATCH_SIZE}

python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR}/all \
  --eval_dir=${TRAIN_DIR}/all \
  --dataset_split_name=val \
  --partition_id=${PARTITION_ID} \
  --partition_num=${PARTITION_NUM} \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL_NAME} \
  --preprocessing_name=inception \
  --eval_image_size=${IMAGE_SIZE} \
  --batch_size=${BATCH_SIZE}

# Check flag.
FLAG="$(python early_stop.py --logdir ${TRAIN_DIR}/all --patience $PATIENCE)"
if [ $FLAG = stop ]
  then
  break
fi
done
