import keras 
import numpy as np
import pandas as pd
import cv2
import os
import json
import pdb
import argparse
import math
import copy
from vis.visualization import visualize_cam, overlay, visualize_activation
from vis.utils.utils import apply_modifications
from shutil import rmtree
import matplotlib.cm as cm
from matplotlib import pyplot as plt

from sklearn import metrics

import keras.backend as K
from keras import activations
from keras.applications.inception_v3 import preprocess_input as inception_pre
from keras.applications.mobilenet import preprocess_input as mobilenet_pre
from keras.applications.resnet50 import preprocess_input as resnet_pre
from keras.applications.densenet import preprocess_input as densenet_pre
from datagenerator import ImageDataGenerator

from utils import load_model

def getCAM(model, image):
	# weights of the final fully-connected layer
	weights = model.layers[-1].get_weights()[0]

	# activation before the last global pooling
	for layer in reversed(model.layers):
		if len(layer.output_shape) > 2:
			break
	function = K.function([model.layers[0].input, K.learning_phase()], [layer.output])
	activation = np.squeeze(function([image, 0])[0])

	# weighted sum of the activation map
	CAM = np.dot(activation, weights)

	return CAM


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--ckpt_path', help = 'Path to the model checkpoint.')
	ap.add_argument('--image_path', help = 'Path to the image to run inference on.')
	ap.add_argument('--bnbox', help = 'Path to the bounding box annotation, if applies.')
	ap.add_argument('--threshold', default = 0.5, help = 'Threshold for displaying the Class Activation Map.')

	args = ap.parse_args()

	model_dir = os.path.dirname(args.ckpt_path)

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

	if args.bnbox is not None:
		annotation = pd.read_csv(args.bnbox)

	image_index = os.path.basename(args.image_path)

	indices = np.where(annotation['Image Index'] == image_index)[0]

	bnbox = {}

	for i in indices:
		disease = annotation['Finding Label'][i]
		x = int(annotation['Bbox [x'][i] + 0.5)
		y = int(annotation['y'][i] + 0.5)
		w = int(annotation['w'][i] + 0.5)
		h = int(annotation['h]'][i] + 0.5)
		bnbox[disease] = [x, y, x + w, y + h]

	image = cv2.imread(args.image_path)

	img = cv2.resize(image, (image_size, image_size))

	img = preprocess_input[model_name](img.astype(np.float32))

	img = np.expand_dims(img, axis = 0)

	predictions = np.squeeze(model.predict(img))

	CAM = getCAM(model, img)

	cv2.namedWindow("ChestXray", cv2.WINDOW_NORMAL)

	for key, value in label_map.items():

		heatmap = CAM[:,:,int(key)]
		heatmap -= heatmap.min()
		heatmap *= 255.0 / heatmap.max()
		heatmap[np.where(heatmap < args.threshold * 255)] *= 0.1

		heatmap = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)

		heatmap = cv2.resize(heatmap, image.shape[:2], cv2.INTER_AREA)

		overlay_img = overlay(heatmap, image, alpha = 0.4)

		cv2.putText(overlay_img, "{}: {:.2%}".format(value, predictions[int(key)]),
					(30,30), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,255,255), 2)

		if value in bnbox.keys():
			box = bnbox[value]
			cv2.rectangle(overlay_img, (box[0], box[1]), (box[2], box[3]),
							color = (0, 180, 0), thickness = 2)

		cv2.imshow("ChestXray", overlay_img)
		cv2.waitKey()

		plt.show()

		print('{}: {:.2%}'.format(value, predictions[int(key)]))

	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()