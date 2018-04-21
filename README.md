# Private Aggregate of Teacher Ensembles for Diagnosing ChestXray Images

This repository contains the Keras implementation of Private Aggregate of Teacher Ensembles (PATE) model by Nicolas Papernot et al. The model trains privacy-preserving student model by transferring knowledge from an ensemble of teacher models trained on siloed datasets. For more information about the model, please refer to the original paper: [arXiv:1610.05755](https://arxiv.org/abs/1610.05755)

## Dataset

The ChestX-ray14 dataset is used to demonstrate the PATE model. The dataset contains 112,120 frontal view thoracic X-ray images collected from 32,717 unique patients with 14 pathology labels (where each sample may have multiple labels) text-mined from radiological reports. For details about the dataset, please refer to the publication: [arXiv:1705.02315](https://arxiv.org/abs/1705.02315)

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



**Partitioning the dataset and generating file lists:** 

```
python generate_filelist.py --
```

**Training teacher models:**

```
bash train_teachers.sh
```

**Evaluating teacher models:**
```
bash evaluate_teachers.sh
```

**Training student model:**
```
python train_student.py
```

**Evaluating student model:**
```
python evaluate_model.py
```

**Run inference:**
```
python inference.py
```


