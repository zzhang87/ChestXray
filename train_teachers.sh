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

#-----------------Load Modules------------------------------------
# module load Tensorflow-GPU/1.2.1-IGB-gcc-4.9.4-Python-3.6.1

#-----------------Define Variables-------------------------------
MODEL_NAME=mobilenet

# Where the pre-trained ResNetV1-50 checkpoint is saved to.

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=/home/zzhang52/Insight/runs/3_teachers_100th
rm -rf $TRAIN_DIR
mkdir $TRAIN_DIR

# Where the dataset is saved to.
DATA_DIR=/home/zzhang52/Insight/CXR/3_part_100th

IMAGE_DIR=/home/zzhang52/Insight/CXR/images/all

PARTITION_NUM=3

NUM_EPOCH=3

BATCH_SIZE=32

OPTIMIZER=adam

LR=0.01


#-----------------------Execute Commands------------------------------

for teacher in `seq 1 $PARTITION_NUM`
do
# Fine-tune only the new layers.
python train_one_teacher.py \
  --data_dir=${DATA_DIR} \
  --image_dir=${IMAGE_DIR} \
  --train_dir=${TRAIN_DIR}/teacher_${teacher} \
  --partition_id=${teacher} \
  --partition_num=${PARTITION_NUM} \
  --batch_size=${BATCH_SIZE} \
  --num_epoch=${NUM_EPOCH} \
  --model_name=${MODEL_NAME} \
  --optimizer=${OPTIMIZER} \
  --initial_lr=${LR}

done
