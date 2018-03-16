import os
import json
import argparse
import numpy as np

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--dir')
	args = ap.parse_args()

	file_list = [x for x in os.listdir(args.dir) if 'test' in x and 'auc' in x]

	best = 0.
	best_epoch = None

	for file in file_list:
		epoch = file.replace('_', '.').split('.')
		epoch = int(epoch[-2])

		with open(os.path.join(args.dir, file), 'r') as f:
			auc_scores = json.load(f)

		mean_auc = np.mean(np.array(auc_scores.values()))
		if mean_auc > best:
			best_epoch = epoch
			best = mean_auc


	print("Epoch with highest mean AUROC: {}".format(best_epoch))

if __name__ == "__main__":
	main()