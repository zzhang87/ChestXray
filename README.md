# Private Aggregate of Teacher Ensembles for Diagnosing ChestXray Images

This repository contains the Keras implementation of Private Aggregate of Teacher Ensembles (PATE) model by Nicolas Papernot et al. The model trains privacy-preserving student model by transferring knowledge from an ensemble of teacher models trained on siloed datasets. For more information about the model, please refer to the original paper: [arXiv:1610.05755](https://arxiv.org/abs/1610.05755)

## Dataset

The ChestX-ray14 (CXR14) dataset is used to demonstrate the PATE model. The dataset contains 112,120 frontal view thoracic X-ray images collected from 32,717 unique patients with 14 pathology labels (where each sample may have multiple labels) text-mined from radiological reports. For details about the dataset, please refer to the publication: [arXiv:1705.02315](https://arxiv.org/abs/1705.02315)

To download the dataset, visit (https://nihcc.app.box.com/v/ChestXray-NIHCC). Download all the zip files and extract the raw images to a common directory.

## Requirements

* Keras
* numpy
* scipy
* scikit-learn
* OpenCV
* Pillow
* h5py
* pandas
* six

## How to run

The repository provides the script to split the dataset into training, validation, and testing sets with any given ratio. The training and validation sets are then evenly partitioned to represent siloed datasets. To train `n` teacher models, the user can create `n+1` partitions and train the teachers using partitions `1` through `n`, and train the student using partition `n+1`. The training scripts accept lists of file names of the input images and use a modified Keras ImageDataGenerator to generate batches of input data with real-time augmentation. The implementation supports using one of four ImageNet pre-trained CNN architectures - MobileNet, Inception, ResNet, and DenseNet. 

**Partitioning the dataset and generating file lists:** 

```
python generate_filelist.py \
		--out_dir path/to/output \
		--data_log path/to/datalog \ #'Data_Entry_2017.csv' from the CXR14 dataset
		--partition n 
```
Use `--help` to see the full list of flags. 

**Training teacher models:**
Modify the variables in `train_teachers.sh` to the appropriate values, and execute:
```
bash train_teachers.sh
```
The script calls `train_one_teacher.py` and trains each teacher sequentially. 

**Evaluating teacher models:**
Modify the variables in `evaluate_teachers.sh` to the appropriate values, and execute:
```
bash evaluate_teachers.sh
```
The script calls `evaluate_model.py` and evaluates each teacher on validation and testing sets sequentially.

**Training student model:**
```
python train_student.py \
		--data_dir path/to/file/lists \
		--image_dir path/to/raw/images \
		--teacher_dir path/to/teachers \
		--student_dir path/to/output 
```

**Evaluating student model:**
```
python evaluate_model.py \
		--data_dir path/to/file/lists \
		--ckpt_path path/to/model/checkpoint \
		--image_dir path/to/raw/images \
		--split_name val/or/test
```

**Run inference:**
Finally, the user can run sample inference using:
```
python inference.py \
		--ckpt_path path/to/checkpoint \
		--image_path path/to/query/image
```
The script will predict the probability of each pathology and overlay the class activation map on the original image.

## Sample outputs


