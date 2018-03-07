import numpy as np
import os
import argparse

import tensorflow as tf

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--logdir', type = str, help = 'Directory to the model logs')

	args = ap.parse_args()

	files = os.listdir(args.logdir)
	events = [f for f in files if 'event' in f]
	events.sort()

	assert(len(events) > 0)

	event = os.path.join(args.logdir, events[-1])

	for e in tf.train.summary_iterator(event):
		for v in e.summary.value:
			if 'Loss' in v.tag:
				print('{}: {}'.format(v.tag, v.simple_value))

if __name__ == "__main__":
	main()