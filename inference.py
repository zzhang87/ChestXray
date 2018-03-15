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
from keras.applications.densenet import preprocess_input as densenet_pre
from datagenerator import ImageDataGenerator

from utils import load_model


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--ckpt_path', help = 'Path to the model checkpoint.')
	ap.add_argument('--image_path', help = 'Path to the image to run inference on.')

	args = ap.parse_args()

	model_dir = os.path.basename(args.ckpt_path)

	with open(os.path.join(model_dir, 'label_map.json'), 'r') as f:
		label_map = json.load(f)

	num_class = len(list(label_map.keys()))

	model, model_config = load_model(model_dir, args.ckpt_path)

	model_name = model_config['model_name']

	if model_name in ['inception']:
		image_size = 299

	else:
		image_size = 224

	preprocess_input = {
		'inception': inception_pre,
		'resnet': resnet_pre,
		'mobilenet': mobilenet_pre,
		'densenet': densenet_pre
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