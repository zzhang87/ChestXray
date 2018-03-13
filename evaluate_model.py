import keras 
import numpy as np
import pandas as pd
import cv2
import os
import json
import PIL
import pdb
import argparse
import math
from shutil import rmtree

from sklearn import metrics

import keras.backend as K
from keras.applications.inception_v3 import preprocess_input as inception_pre
from keras.applications.mobilenet import preprocess_input as mobilenet_pre
from keras.applications.resnet50 import preprocess_input as resnet_pre
from keras.applications.densenet import preprocess_input as densnet_pre
from datagenerator import ImageDataGenerator

from utils import load_filelist, load_model


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--model_dir',
			help = 'Directory to model checkpoints and config. The latest checkpoint will be evaluated.')
	ap.add_argument('--data_dir', help = 'Directory to the file lists and labels.')
	ap.add_argument('--ckpt_path', help = 'Path to a specific checkpoint to evaluate.')
	ap.add_argument('--image_dir', help = 'Directory to the raw images.')
	ap.add_argument('--partition_id', type = int, default = 1,
					help = 'Partition index (1-based).')
	ap.add_argument('--partition_num', type = int, default = 1,
					help = 'Number of partitions.')
	ap.add_argument('--split_name', default = 'val',
					help = "Dataset split to evaluate on. Either 'val' or 'test'.")
	ap.add_argument('--batch_size', type = int, default = 32)

	args = ap.parse_args()

	with open(os.path.join(args.data_dir, 'label_map.json'), 'r') as f:
		label_map = json.load(f)

	num_class = len(list(label_map.keys()))

	model, model_config = load_model(args.model_dir, args.ckpt_path)

	model_name = model_config['model_name']

	if model_name in ['inception']:
		image_size = 299

	elif model_name in ['resnet', 'mobilenet', 'densenet']:
		image_size = 224

	preprocess_input = {
		'inception': inception_pre,
		'resnet': resnet_pre,
		'mobilenet': mobilenet_pre,
		'densenet': densnet_pre
	}

	datagen = ImageDataGenerator(preprocessing_function = preprocess_input[model_name])

	X, Y = load_filelist(args.data_dir, args.split_name, args.partition_id, args.partition_num)

	predictions = model.predict_generator(datagen.flow_from_list(x = X, directory = args.image_dir,
								batch_size = args.batch_size, target_size = (image_size, image_size),
								shuffle = False))

	labels = np.array(Y)

	auc_scores = {}

	for key, value in label_map.items():
		pred = predictions[:,int(key)]
		label = labels[:,int(key)]

		if len(np.unique(label)) == 1:
			auc_scores[value] = np.nan

		else:
			auc_scores[value] = metrics.roc_auc_score(label, pred)

	with open(os.path.join(model_config['model_dir'],
				'{}_auc_scores_{:03}.json'.format(args.split_name, model_config['epoch'])), 'w') as f:
		json.dump(auc_scores, f)


if __name__ == "__main__":
	main()