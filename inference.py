import keras 
import numpy as np
import pandas as pd
import cv2
import os
import json
import pdb
import argparse
import math
from vis.visualization import visualize_cam, overlay
from shutil import rmtree

from sklearn import metrics

import keras.backend as K
from keras.applications.inception_v3 import preprocess_input as inception_pre
from keras.applications.mobilenet import preprocess_input as mobilenet_pre
from keras.applications.resnet50 import preprocess_input as resnet_pre
from keras.applications.densenet import preprocess_input as densnet_pre
from datagenerator import ImageDataGenerator

from utils import load_model


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--model_dir',
			help = 'Directory to model checkpoints and config. The latest checkpoint will be evaluated.')
	ap.add_argument('--ckpt_path', help = 'Path to a specific checkpoint to evaluate.')
	ap.add_argument('--image_dir', help = 'Directory to the raw images.')

	args = ap.parse_args()

	with open(os.path.join(args.model_dir, 'label_map.json'), 'r') as f:
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

	image = cv2.imread(args.image_dir)

	img = cv2.resize(image, (image_size, image_size))

	img = preprocess_input[model_name](img.astype(np.float32))

	img = np.expand_dims(img, axis = 0)

	predictions = np.squeeze(model.predict(img))

	cv2.namedWindow("ChestXray", cv2.WINDOW_NORMAL)

	for key, value in label_map.items():

		heatmap = visualize_cam(model, layer_idx = -1, filter_indices = int(key), seed_input = img)

		heatmap = cv2.resize(heatmap, image.shape[:2])

		overlay_img = overlay(heatmap, image, alpha = 0.25)

		cv2.putText(overlay_img, "{}: {:.2%}".format(value, predictions[int(key)]),
					(30,30), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,255,255), 2)
		cv2.imshow("ChestXray", overlay_img)
		cv2.waitKey()

		print('{}: {:.2%}'.format(value, predictions[int(key)]))

	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()