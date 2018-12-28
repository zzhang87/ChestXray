import os
import json
import argparse
import numpy as np

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--dir')
	ap.add_argument('--split_name', default = 'val')
	args = ap.parse_args()

	file_list = [x for x in os.listdir(args.dir) if args.split_name in x and 'auc' in x]

	best = 0.
	best_epoch = None

	for file in file_list:
		epoch = file.replace('_', '.').split('.')
		epoch = int(epoch[-2])

		with open(os.path.join(args.dir, file), 'r') as f:
			auc_scores = json.load(f)

		auc = auc_scores['pneumothorax']['auc']
		if auc > best:
			best_epoch = epoch
			best = auc


	print("Epoch {} has highest AUROC: {}".format(best_epoch, best))

if __name__ == "__main__":
	main()
