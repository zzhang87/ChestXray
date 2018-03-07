import os
import io
import numpy as np
import cv2
import PIL.Image
import argparse

import tensorflow as tf

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("-f", "--file", type = str,
					help = "path to the TFrecord file")
	ap.add_argument("-s", "--skip", type = int,
					help = "sample every kth entry", default = 1)

	args = ap.parse_args()

	it = tf.python_io.tf_record_iterator(path = args.file)

	for idx, record in enumerate(it):
		if idx % args.skip != 0:
			continue

		example = tf.train.Example()
		example.ParseFromString(record)

		height = int(example.features.feature['image/height'].int64_list.value[0])
		width = int(example.features.feature['image/width'].int64_list.value[0])

		labels = map(int, example.features.feature['image/class/label'].int64_list.value)

		encoded_img = example.features.feature['image/encoded'].bytes_list.value[0]

		encoded_img_io = io.BytesIO(encoded_img)
		img = PIL.Image.open(encoded_img_io)

		img = np.array(img)

		# img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

		cv2.imshow("image", img)

		print(labels)

		key = cv2.waitKey()

		if key == 27:
			break

		if key == 32:
			continue


	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()