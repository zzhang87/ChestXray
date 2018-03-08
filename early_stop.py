import numpy as np
import os
import argparse

import tensorflow as tf

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--logdir', type = str, help = 'Directory to the model logs')
	ap.add_argument('--patience', type = int, help = 'Patience parameter')

	args = ap.parse_args()

	if os.path.exists(os.path.join(args.logdir, 'val_loss.txt')):
		losses = np.loadtxt(os.path.join(args.logdir, 'val_loss.txt'))
		if losses.ndim < 1:
			losses = np.expand_dims(losses, 0)
		losses = list(losses)
	else:
		losses = []

	files = os.listdir(args.logdir)
	events = [f for f in files if 'event' in f]
	events.sort()

	assert(len(events) > 0)

	event = os.path.join(args.logdir, events[-1])

	for e in tf.train.summary_iterator(event):
		for v in e.summary.value:
			if 'Loss' in v.tag:
				losses.append(v.simple_value)

	arr = np.array(losses, dtype = np.float32)
	np.savetxt(os.path.join(args.logdir, 'val_loss.txt'), arr)

	diff = np.diff(arr)

	if len(arr) <= args.patience:
		print('continue')

	else:
		stop = True

		for i in diff[-args.patience:]:
			stop &= i > 0

		if stop:
			print('stop')

		else:
			print('continue')


if __name__ == "__main__":
	main()