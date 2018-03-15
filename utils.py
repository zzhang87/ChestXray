import keras 
import numpy as np
import pandas as pd
import cv2
import os
import json
import pdb

import tensorflow as tf

import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense
from keras.utils.generic_utils import CustomObjectScope
from keras.optimizers import SGD, Adam, RMSprop
from keras.applications.inception_v3 import InceptionV3
from keras.applications.mobilenet import MobileNet, relu6, DepthwiseConv2D
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121
from datagenerator import ImageDataGenerator

def create_model(model_config, image_size, label_map, loss = None):
	model_map = {
		'inception': InceptionV3,
		'mobilenet': MobileNet,
		'densenet': DenseNet121,
		'resnet': ResNet50
	}

	num_class = len(list(label_map.keys()))
	base_model = model_map[model_config['model_name']](include_top = False,
					input_shape = (image_size, image_size, 3), pooling = 'avg')
	x = base_model.output
	# x = Dense(base_model.output_shape[1], activation = 'relu')(x)
	predictions = Dense(num_class, activation = 'sigmoid')(x)

	model = Model(inputs = base_model.input, outputs = predictions)

	opt = model_config['optimizer']
	if opt == 'SGD':
		optimizer = SGD(lr = model_config['initial_lr'], decay = 1e-6, momentum = 0.9, nesterov = True)
	elif opt == 'adam':
		optimizer = Adam(lr = model_config['initial_lr'], decay = 1e-6)
	else:
		optimizer = RMSprop(lr = model_config['initial_lr'], decay = 1e-6)

	# metrics = {value: AUC(int(key)) for key, value in label_map.items()}

	metrics = [AUC(i) for i in range(num_class)]
	metrics.append(mean_AUC(num_class))

	if loss is None:
		model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = metrics)
	else:
		model.compile(loss = loss, optimizer = optimizer, metrics = metrics)

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


def load_model(model_dir, ckpt_path = None):
	assert(model_dir is not None or ckpt_path is not None)

	if ckpt_path is not None:
		model_dir = os.path.dirname(ckpt_path)

	else:
		ckpts = [x for x in os.listdir(model_dir) if 'hdf5' in x]
		ckpts.sort()
		ckpt_path = os.path.join(model_dir, ckpts[-1])

	with open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
		model_config = json.load(f)

	basename = os.path.basename(ckpt_path)

	epoch = basename.replace('-', '.').split('.')
	epoch = int(epoch[1])

	model_config['epoch'] = epoch
	model_config['model_dir'] = model_dir

	custom_objects = {'auc': auc, 'mauc': auc}

	if model_config['model_name'] == 'mobilenet':
		custom_objects['relu6'] = relu6
		custom_objects['DepthwiseConv2D'] = DepthwiseConv2D

	model = keras.models.load_model(ckpt_path, custom_objects = custom_objects)

	return model, model_config


def auc(labels, predictions):
	score, up_opt = tf.metrics.auc(labels, predictions)
	K.get_session().run(tf.local_variables_initializer())
	with tf.control_dependencies([up_opt]):
		score = tf.identity(score)
	return score


def aggregate_teachers(predictions):
	return np.mean(predictions, axis = 0)


def weighted_binary_crossentropy(alpha):

	def weighted_loss(y_true, y_pred):
		num_class = tf.shape(y_pred)[1]
		hard = keras.losses.binary_crossentropy(y_true[:,:num_class], y_pred)
		soft = keras.losses.binary_crossentropy(y_true[:,num_class:], y_pred)

		return alpha * hard + (1 - alpha) * soft

	def loss(y_true, y_pred):
		return keras.losses.binary_crossentropy(y_true, y_pred)
		
	def my_loss(y_true, y_pred):
		return tf.cond(tf.equal(tf.shape(y_true)[1], tf.shape(y_pred)[1]),
						lambda: loss(y_true, y_pred), lambda: weighted_loss(y_true, y_pred))


	return my_loss


def calc_weights(labels):
	labels = np.array(labels).astype(np.float32)

	freq = np.mean(labels, axis = 0)

	inverse = 1. / freq

	weights = {key: value for key, value in enumerate(inverse)}

	return weights

