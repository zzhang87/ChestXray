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
		# labels = np.zeros(len(class_name_to_ids.keys()), dtype = int)
		labels = np.zeros((1), dtype = int)
		if "Pneumothorax" in findings:
			labels[0] = 1

		files.append((image, labels))



def main():

	ap = argparse.ArgumentParser()
	ap.add_argument("--out_dir", type = str, required = True,
					help = "Directory to store the output dataset")
	ap.add_argument("--data_log", type = str, required = True,
					help = "Directory to the data log file")
	ap.add_argument("--image_dir", type = str,
					help = "Directory to the image files")
	ap.add_argument("--val_ratio", type = float, default = 0.1,
					help = "Percentage of data reserved for validation (in decimal)")
	ap.add_argument("--test_ratio", type = float, default = 0.2,
					help = "Percentage of data reserved for test (in decimal)")
	ap.add_argument("--skip", type = int, default = 1,
					help = "Sample every other k-th entry in the dataset")
	ap.add_argument("-p", "--partition", type = int, default = 1,
					help = "Number of partitions to create")
					
	args = ap.parse_args()

	if args.image_dir is not None:
		image_dir = os.path.abspath(args.image_dir)
		shards = os.listdir(image_dir)
		subdir = {}

		for shard in shards:
			idx = int(os.path.basename(shard))
			shard_dir = os.path.join(image_dir, shard)
			subdir[idx] = set(os.listdir(shard_dir))

	log = pd.read_csv(args.data_log)
	patient_map = {}

	class_name_to_ids = {}

	num_samples = 0
	for i, patient_id in enumerate(log['Patient ID']):
		if i % args.skip != 0:
			continue
			
		image = log['Image Index'][i]
		if args.image_dir is not None:
			for idx, files in subdir.items():
				if image in files:
					image = os.path.join(str(idx), image)
					break

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

	# label_map = {int(value) : key for key, value in class_name_to_ids.items()}
	label_map = {0: "pneumothorax"}

	with open(os.path.join(args.out_dir, "label_map.json"), 'w') as f:
		json.dump(label_map, f)

	for key in train_list.keys():
		save_list(train_list[key], 'train', args.out_dir, key, args.partition)

		save_list(val_list[key], 'val', args.out_dir, key, args.partition)

	save_list(test_list, 'test', args.out_dir, 0, 1)

if __name__ == "__main__":
	main()
