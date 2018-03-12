import keras 
import numpy as np
import pandas as pd
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
from keras.applications.inception_v3 import InceptionV3, preprocess_input as inception_pre
from keras.applications.mobilenet import MobileNet, preprocess_input as mobilenet_pre
from keras.applications.resnet50 import ResNet50, preprocess_input as resnet_pre
from keras.applications.densenet import DenseNet121, preprocess_input as densnet_pre
from datagenerator import ImageDataGenerator

def create_model(name, image_size, label_map):
	model_map = {
		'inception': InceptionV3,
		'mobilenet': MobileNet,
		'densenet': DenseNet121,
		'resnet': ResNet50
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
	metrics.append(mean_AUC(num_class))

	model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = metrics)

	return model

def load_filelist(directory, split_name, partition_id, partition_num):
	path = os.path.join(directory, '{}_{:03}-of-{:03}.csv'.format(split_name, partition_id, partition_num))
	df = pd.read_csv(path, delimiter = '\t', header = None, names = ['image', 'label'])

	labels = [map(float, label[1:-1].split(' ')) for label in df['label']]

	return df['image'].tolist(), labels

def AUC(index):
	def auc(labels, predictions):
		score, up_opt = tf.metrics.auc(labels[:,index], predictions[:,index])
		K.get_session().run(tf.local_variables_initializer())
		with tf.control_dependencies([up_opt]):
			score = tf.identity(score)
		return score
	return auc

def mean_AUC(num_class):
	def mauc(labels, predictions):
		scores = []
		opts = []
		for i in range(num_class):
			score, up_opt = tf.metrics.auc(labels[:,i], predictions[:,i])
			scores.append(score)
			opts.append(up_opt)

		score = tf.add_n(scores) / float(num_class)
		K.get_session().run(tf.local_variables_initializer())
		with tf.control_dependencies(opts):
			score = tf.identity(score)
		return score
	return mauc

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--data_dir', help = 'Directory to the file lists and labels.')
	ap.add_argument('--image_dir', help = 'Directory to the raw images.')
	ap.add_argument('--partition_id', type = int, default = 1,
					help = 'Partition index (1-based).')
	ap.add_argument('--partition_num', type = int, default = 1,
					help = 'Number of partitions.')
	ap.add_argument('--train_dir', help = 'Directory the trained model and events will be saved to.')
	ap.add_argument('--model_name', default = 'inception',
					help = 'Network architecture to use. One of inception, resnet, densenet, mobilenet.')
	ap.add_argument('--batch_size', type = int, default = 32)
	ap.add_argument('--num_epoch', type = int, default = 1)

	args = ap.parse_args()

	if os.path.isdir(args.train_dir):
		rmtree(args.train_dir)

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

	X_train, Y_train = load_filelist(args.data_dir, 'train', args.partition_id, args.partition_num)

	X_val, Y_val = load_filelist(args.data_dir, 'val', args.partition_id, args.partition_num)

	preprocess_input = {
		'inception': inception_pre,
		'resnet': resnet_pre,
		'mobilenet': mobilenet_pre,
		'densenet': densnet_pre
	}

	gen_train = ImageDataGenerator(rotation_range = 10,
								width_shift_range = 0.1,
								height_shift_range = 0.1,
								zoom_range = 0.1,
								fill_mode = 'constant',
								cval = 0,
								horizontal_flip = True,
								preprocessing_function = preprocess_input[args.model_name])

	gen_val = ImageDataGenerator(preprocessing_function = preprocess_input[args.model_name])

	# for x, y in gen_train.flow_from_list(x=X_train, y=Y_train, directory=args.image_dir,
	# 						batch_size = args.batch_size, target_size=(image_size,image_size)):
	# 	for i in range(x.shape[0]):
	# 		img = x[i].astype(np.uint8)
	# 		label = y[i]

	# 		cv2.imshow("show", img)
	# 		cv2.waitKey(600)
	# 		cv2.destroyAllWindows()
	# 		print(label)

	model = create_model(args.model_name, image_size, label_map)

	tensorbard = TensorBoard(args.train_dir)
	reducelr = ReduceLROnPlateau(monitor = 'loss', factor = 0.9, patience = 5, mode = 'min')
	earlystop = EarlyStopping(monitor = 'val_mauc', min_delta = 1e-4,
								patience = args.num_epoch / 10, mode = 'max')
	ckpt = ModelCheckpoint(os.path.join(args.train_dir, 'weights.{epoch:03d}-{val_mauc:.2f}.hdf5'),
								monitor = 'val_mauc', save_best_only = True, mode = 'max')

	history = model.fit_generator(gen_train.flow_from_list(x=X_train, y=Y_train, directory=args.image_dir,
							batch_size = args.batch_size, target_size=(image_size,image_size)), epochs = args.num_epoch,
							steps_per_epoch = math.ceil(len(X_train) / float(args.batch_size)),
							validation_data = gen_val.flow_from_list(x=X_val, y=Y_val, directory=args.image_dir,
							batch_size = args.batch_size, target_size=(image_size,image_size)),
							validation_steps = math.ceil(len(X_val) / float(args.batch_size)),
							verbose = 2, callbacks = [tensorbard, reducelr, earlystop, ckpt])


if __name__ == "__main__":
	main()
