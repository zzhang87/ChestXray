import os 
import random
import argparse
import sys
from shutil import rmtree
import math
import json
import csv
import cv2
import pandas as pd
import numpy as np
from random import shuffle
from datasets import dataset_utils
import tensorflow as tf

def convert_dataset(files, split_name, input_dir, output_dir, partition_id, partition_num):
    assert split_name in ['train', 'val', 'test']

    with tf.Graph().as_default():
        decode_jpeg_data = tf.placeholder(dtype=tf.string)
        decode_jpeg = tf.image.decode_image(decode_jpeg_data, channels=3)

        with tf.Session('') as sess:
        	output_path = os.path.join(output_dir,
        		'{}_{:03}-of-{:03}.tfrecord'.format(split_name, partition_id + 1, partition_num))

        	tfrecord_writer = tf.python_io.TFRecordWriter(output_path)
        	f = open(os.path.join(output_dir,
        		'{}_{:03}-of-{:03}.txt'.format(split_name, partition_id + 1, partition_num)), 'w')
        	for i, record in enumerate(files):
        		sys.stdout.write('\r>> Converting image {}/{} {} partition {}/{}'.format(
        						i + 1, len(files), split_name, partition_id + 1, partition_num))
        		sys.stdout.flush()

        		image, label = record

        		f.write(image + '\n')

        		image_data = tf.gfile.FastGFile(os.path.join(input_dir, image), 'rb').read()
        		image = sess.run(decode_jpeg, feed_dict = {decode_jpeg_data: image_data})
        		height, width, channels = image.shape[0], image.shape[1], image.shape[2]
        		example = dataset_utils.image_to_tfexample(image_data, b'png', height, width, list(label))
        		tfrecord_writer.write(example.SerializeToString())

        	tfrecord_writer.close()

    sys.stdout.write('\n')
    sys.stdout.flush()
    f.close()

def save_list(filelist, split_name, out_dir, partition_id, partition_num):
	filename = '{}_{:03}-of-{:03}.csv'.format(split_name, partition_id + 1, partition_num)
	filename = os.path.join(out_dir, filename)

	with open(filename, 'w') as f:
		writer = csv.writer(f, delimiter = '\t', lineterminator = '\n')
		writer.writerows(filelist)

	f.close()


def extract(files, record, class_name_to_ids):
	for entry in record:
		image = entry[0]
		findings = entry[1]
		labels = np.zeros(len(class_name_to_ids.keys()), dtype = int)
		for label in findings:
			labels[class_name_to_ids[label]] = 1

		files.append((image, labels))



def main():

	ap = argparse.ArgumentParser()
	ap.add_argument("--out_dir", type = str, required = True,
					help = "Directory to store the output dataset")
	ap.add_argument("--data_log", type = str, required = True,
					help = "Directory to the data log file")
	ap.add_argument("--val_ratio", type = float, default = 0.1,
					help = "Percentage of data reserved for validation (in decimal)")
	ap.add_argument("--test_ratio", type = float, default = 0.2,
					help = "Percentage of data reserved for test (in decimal)")
	ap.add_argument("--skip", type = int, default = 1,
					help = "Sample every other k-th entry in the dataset")
	ap.add_argument("-p", "--partition", type = int, default = 1,
					help = "Number of partitions to create")
					
	args = ap.parse_args()

	log = pd.read_csv(args.data_log)
	patient_map = {}

	class_name_to_ids = {}

	num_samples = 0
	for i, patient_id in enumerate(log['Patient ID']):
		if i % args.skip != 0:
			continue
			
		image = log['Image Index'][i]
		labels = log['Finding Labels'][i].split('|')

		if patient_id not in patient_map.keys():
			patient_map[patient_id] = []

		patient_map[patient_id].append((image,labels))

		for label in labels:
			if label not in class_name_to_ids.keys():
				class_name_to_ids[label] = len(class_name_to_ids.keys())
				
		num_samples += 1

	
	num_val = int(num_samples * args.val_ratio)
	num_test = int(num_samples * args.test_ratio)
	num_train = num_samples - num_val - num_test
	
	num_val_per_partition = int(math.ceil(float(num_val) / args.partition))
	num_train_per_partition = int(math.ceil((num_samples - num_test - num_val_per_partition*args.partition)
								/ float(args.partition)))

	val_list = {i : [] for i in range(args.partition)}
	train_list = {i : [] for i in range(args.partition)}
	test_list = []
	
	print(num_samples, num_train, num_val, num_test)
	print(sum([len(x) for x in patient_map.values()]))
	print(num_train_per_partition, num_val_per_partition)

	records = list(patient_map.values())
	shuffle(records)

	for record in records:

		if min([len(x) for x in val_list.values()]) < num_val_per_partition:
			for key, value in val_list.items():
				if len(value) >= num_val_per_partition:
					continue
				extract(val_list[key], record, class_name_to_ids)
				break

		elif len(test_list) < num_test:
			extract(test_list, record, class_name_to_ids)

		else:
			for key, value in train_list.items():
				if len(value) >= num_train_per_partition:
					continue
				extract(train_list[key], record, class_name_to_ids)
				break

	num_samples = {}
	num_samples["train"] = {key : len(value) for key, value in train_list.items()}
	num_samples["val"] = {key : len(value) for key, value in val_list.items()}
	num_samples["test"] = len(test_list)

	if os.path.isdir(args.out_dir):
		rmtree(args.out_dir)

	os.mkdir(args.out_dir)

	with open(os.path.join(args.out_dir, "num_samples.json"), 'w') as f:
		json.dump(num_samples, f)

	label_map = {int(value) : key for key, value in class_name_to_ids.items()}

	with open(os.path.join(args.out_dir, "label_map.json"), 'w') as f:
		json.dump(label_map, f)

	for key in train_list.keys():
		save_list(train_list[key], 'train', args.out_dir, key, args.partition)

		save_list(val_list[key], 'val', args.out_dir, key, args.partition)

	save_list(test_list, 'test', args.out_dir, 0, 1)

if __name__ == "__main__":
	main()
