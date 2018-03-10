import os
import io
import numpy as np
import pandas as pd
import cv2
import PIL.Image
import argparse
import json

import tensorflow as tf

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("-f", "--file", type = str,
					help = "path to the TFrecord file")
	ap.add_argument("-i", "--images", type =str,
					help = "path to the image indices")
	ap.add_argument("--data_dir", type = str,
					help = "directory to the image files")
	ap.add_argument("--label_map", type = str,
					help = "path to the label map")
	ap.add_argument("-l", "--log", type = str,
					help = "path to the data log")
	ap.add_argument("-s", "--skip", type = int,
					help = "sample every kth entry", default = 1)

	args = ap.parse_args()

	with open(args.images, 'r') as f:
		images = f.read().splitlines()

	log = pd.read_csv(args.log)

	with open(args.label_map, 'r') as f:
		label_map = json.load(f)

	it = tf.python_io.tf_record_iterator(path = args.file)

	for idx, record in enumerate(it):
		if idx % args.skip != 0:
			continue

		sample = log[log["Image Index"].str.match(images[idx])]
		findings = sample["Finding Labels"].values[0].split("|")

		example = tf.train.Example()
		example.ParseFromString(record)

		height = int(example.features.feature['image/height'].int64_list.value[0])
		width = int(example.features.feature['image/width'].int64_list.value[0])

		labels = map(int, example.features.feature['image/class/label'].int64_list.value)

		_labels = []
		for i, label in enumerate(labels):
			if label == 0:
				continue
			_labels.append(label_map[str(i)])

		assert(findings.sort() == _labels.sort())

		encoded_img = example.features.feature['image/encoded'].bytes_list.value[0]

		encoded_img_io = io.BytesIO(encoded_img)
		img = PIL.Image.open(encoded_img_io)

		img = np.array(img)

		img3d = np.tile(img[..., None], 3)

		_img = cv2.imread(os.path.join(args.data_dir, images[idx]))

		if not np.all(img3d == _img):
			cv2.imshow("TFrecord", img)
			cv2.waitKey()
			cv2.imshow("Read", _img)
			cv2.waitKey()
			cv2.destroyAllWindows()	

if __name__ == "__main__":
	main()