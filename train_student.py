import keras 
import numpy as np
import pandas as pd
import cv2
import os
import json
import pdb
import argparse
import math
from shutil import rmtree
import tensorflow as tf

import keras.backend as K
from keras.callbacks import ProgbarLogger, TensorBoard, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.applications.inception_v3 import preprocess_input as inception_pre
from keras.applications.mobilenet import preprocess_input as mobilenet_pre
from keras.applications.resnet50 import preprocess_input as resnet_pre
from keras.applications.densenet import preprocess_input as densenet_pre
from datagenerator import ImageDataGenerator

from utils import load_filelist, load_model, create_model, aggregate_teachers,
					weighted_binary_crossentropy, calc_weights

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--data_dir', help = 'Directory to the file lists.')
	ap.add_argument('--image_dir', help = 'Directory to the raw images.')
	ap.add_argument('--teacher_dir', help = 'Directory to the trained teachers.')
	ap.add_argument('--student_dir', help = 'Directory the trained student and events will be saved to.')
	ap.add_argument('--model_name', default = 'inception',
					help = 'Network architecture to use. One of inception, resnet, densenet, or mobilenet.')
	ap.add_argument('--batch_size', type = int, default = 32)
	ap.add_argument('--num_epoch', type = int, default = 1)
	ap.add_argument('--optimizer', default = 'SGD',
					help = 'Optimizer to train the model. One of SGD, adam, or rmsprop.')
	ap.add_argument('--initial_lr', type = float, default = 1e-2, help = 'Initial learning rate.')
	ap.add_argument('--partition_id', type = int, default = 1,
					help = 'Partition index (1-based).')
	ap.add_argument('--partition_num', type = int, default = 1,
					help = 'Number of partitions.')
	ap.add_argument('--alpha', type = float, default = 0.25,
					help = """Weight factor for losses between true label and aggregated prediction.
						(1 for true label and 0 for aggregated prediction.""")
	ap.add_argument('--weight_loss', action = 'store_true')

	args = ap.parse_args()

	if os.path.isdir(args.student_dir):
		rmtree(args.student_dir)

	os.mkdir(args.student_dir)

	preprocess_input = {
		'inception': inception_pre,
		'resnet': resnet_pre,
		'mobilenet': mobilenet_pre,
		'densenet': densenet_pre
	}

	image_size = {
		'inception': 299,
		'resnet': 224,
		'mobilenet': 224,
		'densenet': 224
	}

	teacher_dir = [x for x in os.listdir(args.teacher_dir) if 'teacher' in x]
	num_teachers = len(teacher_dir)
	teachers = []

	for directory in teacher_dir:
		directory = os.path.join(args.teacher_dict, directory)
		ckpts = [x for x in os.listdir(directory) if 'hdf5' in x]
		ckpts.sort()
		teacher, config = load_model(directory, ckpts[-1])
		teacher_dict = {}
		teacher_dict['model'] = teacher
		model_name = config['model_name']
		teacher_dict['model_name'] = model_name
		teacher_dict['image_size'] = image_size[model_name]
		teacher_dict['datagen'] = ImageDataGenerator(preprocessing_function = preprocess_input[model_name])

		teachers.append(teacher_dict)

	student_config = {'model_name': args.model_name,
					'optimizer': args.optimizer,
					'initial_lr': args.initial_lr}

	with open(os.path.join(args.student_dir, 'model_config.json'), 'w') as f:
		json.dump(student_config, f)

	with open(os.path.join(args.data_dir, 'label_map.json'), 'r') as f:
		label_map = json.load(f)

	with open(os.path.join(args.student_dir, 'label_map.json'), 'w') as f:
		json.dump(label_map, f)

	num_class = len(list(label_map.keys()))

	X_train, Y_train = load_filelist(args.data_dir, 'train', args.partition_id, args.partition_num)

	X_val, Y_val = load_filelist(args.data_dir, 'val', args.partition_id, args.partition_num)

	datagetn_aug = ImageDataGenerator(rotation_range = 10,
								width_shift_range = 0.1,
								height_shift_range = 0.1,
								zoom_range = 0.1,
								fill_mode = 'constant',
								cval = 0,
								horizontal_flip = True,
								preprocessing_function = preprocess_input[args.model_name])

	datagen = ImageDataGenerator(preprocessing_function = preprocess_input[args.model_name])

	Y = np.empty((num_teachers, len(Y_train), num_class))
	for i, teacher in enumerate(teachers):
		model = teacher['model']
		generator = teacher['datagen']
		size = teacher['image_size']
		Y[i] = model.predict_generator(generator.flow_from_list(x = X_train, directory = args.image_dir,
								batch_size = args.batch_size, target_size = (size, size),
								shuffle = False))

	for teacher in teachers:
		del teacher
	del teachers

	Y = aggregate_teachers(Y)

	Y_train = np.concatenate((np.array(Y_train), Y), axis = -1)

	loss = weighted_binary_crossentropy(args.alpha)

	student = create_model(student_config, image_size[args.model_name], label_map, loss)

	class_weight = None
	if args.weight_loss:
		class_weight = calc_weights(Y_train)

	tensorbard = TensorBoard(args.student_dir)
	reducelr = ReduceLROnPlateau(monitor = 'loss', factor = 0.9, patience = 5, mode = 'min')
	earlystop = EarlyStopping(monitor = 'val_mauc', min_delta = 1e-3,
								patience = max(5, args.num_epoch / 10), mode = 'max')
	ckpt = ModelCheckpoint(os.path.join(args.student_dir, 'weights.{epoch:03d}-{val_mauc:.2f}.hdf5'),
								monitor = 'val_mauc', save_best_only = True, mode = 'max')

	size = image_size[args.model_name]
	student.fit_generator(datagetn_aug.flow_from_list(x = X_train, y = Y_train, directory = args.image_dir,
							batch_size = args.batch_size, target_size = (size, size)), epochs = args.num_epoch,
							steps_per_epoch = math.ceil(len(X_train) / float(args.batch_size)),
							validation_data = datagen.flow_from_list(x=X_val, y=Y_val, directory=args.image_dir,
							batch_size = args.batch_size, target_size=(size, size)),
							validation_steps = math.ceil(len(X_val) / float(args.batch_size)),
							class_weight = class_weight, verbose = 2,
							callbacks = [tensorbard, reducelr, earlystop, ckpt])


if __name__ == "__main__":
	main()