import keras 
import numpy as np
import cv2
import os
import json
import io
import PIL
import pdb
import argparse
import math
from shutil import rmtree
import pandas as pd
import tensorflow as tf

import keras.backend as K
from keras.callbacks import ProgbarLogger, TensorBoard, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import SGD
from keras.applications.inception_v3 import InceptionV3
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.preprocessing.image import ImageDataGenerator

def create_model(name, image_size, label_map):
	model_map = {
		'inception': InceptionV3,
		'mobilenet': MobileNet
	}

	num_class = len(list(label_map.keys()))
	base_model = model_map[name](include_top = False, input_shape = (image_size, image_size, 3), pooling = 'avg')
	x = base_model.output
	# x = Dense(base_model.output_shape[1], activation = 'relu')(x)
	predictions = Dense(num_class, activation = 'sigmoid')(x)

	model = Model(inputs = base_model.input, outputs = predictions)

	opt = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)

	# metrics = {value: AUC(int(key)) for key, value in label_map.items()}

	metrics = [AUC(i) for i in range(num_class)]

	model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = metrics)

	return model

def load_input(data, num_samples, num_class, image_size):
	it = tf.python_io.tf_record_iterator(path = data)

	X = np.empty((num_samples, image_size, image_size, 3))
	Y = np.empty((num_samples, num_class))

	valid = 0

	for record in it:
		example = tf.train.Example()
		example.ParseFromString(record)
		labels = map(int, example.features.feature['image/class/label'].int64_list.value)

		encoded_img = example.features.feature['image/encoded'].bytes_list.value[0]

		encoded_img_io = io.BytesIO(encoded_img)
		img = PIL.Image.open(encoded_img_io)

		img = np.array(img)

		if (img.ndim != 2):
			continue

		img3d = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
		img3d = cv2.resize(img3d, (image_size, image_size))

		X[valid] = img3d
		Y[valid] = np.array(labels)
		valid += 1

	return X[:valid], Y[:valid]

def AUC(index):
	def auc(labels, predictions):
		score, up_opt = tf.metrics.auc(labels[:,index], predictions[:,index])
		K.get_session().run(tf.local_variables_initializer())
		with tf.control_dependencies([up_opt]):
			score = tf.identity(score)
		return score
	return auc

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--data_dir')
	ap.add_argument('--split_name', default = 'train')
	ap.add_argument('--image_dir')
	ap.add_argument('--partition_id', type = int, default = 1)
	ap.add_argument('--partition_num', type = int, default = 1)
	ap.add_argument('--train_dir')
	ap.add_argument('--model_name', default = 'inception')
	ap.add_argument('--batch_size', type = int, default = 32)
	ap.add_argument('--num_epoch', type = int, default = 1)

	args = ap.parse_args()

	if os.path.isdir(args.train_dir):
		rmtree(args.train_dir)

	else:
		os.mkdir(args.train_dir)

	if args.model_name in ['inception']:
		image_size = 299

	elif args.model_name in ['resnet', 'mobilenet', 'densenet']:
		image_size = 224

	else:
		raise(Error('Unrecognized model name'))

	with open(os.path.join(args.data_dir, 'label_map.json'), 'r') as f:
		label_map = json.load(f)

	num_class = len(list(label_map.keys()))

	with open(os.path.join(args.data_dir, 'num_samples.json'), 'r') as f:
		num_samples = json.load(f)

	tfrecord = os.path.join(args.data_dir,
			'{}_{:03}-of-{:03}.tfrecord'.format(args.split_name, args.partition_id, args.partition_num))

	X, Y = load_input(tfrecord, num_samples[args.split_name][str(args.partition_id - 1)],
						num_class, image_size)

	tfrecord = os.path.join(args.data_dir,
			'val_{:03}-of-{:03}.tfrecord'.format(args.partition_id, args.partition_num))

	X_val, Y_val = load_input(tfrecord, num_samples['val'][str(args.partition_id - 1)],
						num_class, image_size)

	gen_train = ImageDataGenerator(rotation_range = 10,
								width_shift_range = 0.1,
								height_shift_range = 0.1,
								zoom_range = 0.1,
								fill_mode = 'constant',
								cval = 0,
								horizontal_flip = True,
								preprocessing_function = preprocess_input)

	gen_val = ImageDataGenerator(preprocessing_function = preprocess_input)

	model = create_model(args.model_name, image_size, label_map)

	tensorbard = TensorBoard(args.train_dir)
	reducelr = ReduceLROnPlateau(monitor = 'loss', factor = 0.8, patience = 5, mode = 'min')
	earlystop = EarlyStopping(monitor = 'val_loss', min_delta = 1e-6, patience = 10, mode = 'min')
	ckpt = ModelCheckpoint(os.path.join(args.train_dir, '{epoch:02d}-{val_loss:.2f}.hdf5'),
				monitor = 'val_loss', save_best_only = True, mode = 'min')

	history = model.fit_generator(gen_train.flow(X, Y, batch_size = args.batch_size), epochs = args.num_epoch,
							steps_per_epoch = math.ceil(X.shape[0] / float(args.batch_size)),
							validation_data = gen_val.flow(X_val, Y_val, batch_size = args.batch_size),
							validation_steps = math.ceil(X_val.shape[0] / float(args.batch_size)),
							verbose = 2,
							callbacks = [tensorbard, reducelr, earlystop, ckpt])


if __name__ == "__main__":
	main()
