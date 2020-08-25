import json
import os
import numpy as np

from scipy.special import softmax

def get_shannon_entropy(logits):
	np_logits = np.array(logits)
	np_softmax = softmax(np_logits, axis=1)
	np_softmax[np_softmax==0] = 1
	entropy = -np.sum(np_softmax * np.log2(np_softmax), axis=1)
	return list(entropy)

def split_correct(logits, y_test):
	np_logits = np.array(logits)
	np_softmax = softmax(np_logits, axis=1)
	y_test = np.array(y_test)

	data = {}
	correct_logits = np_logits[np_softmax.argmax(axis=1)==y_test]
	data['correct'] = list(correct_logits)
	incorrect_logits = np_logits[np_softmax.argmax(axis=1)!=y_test]
	data['incorrect'] = list(incorrect_logits)
	return data

if __name__ == "__main__":

	DIR_PATH = os.getcwd()
	FILE_NAMES = ["no-at-ensemble-logits.json", "PGD-single-logits.json", "PGD-ensemble-logits.json"]

	with open(os.path.join(DIR_PATH, "y_test.json"), 'r') as y_test_file:
		Y_TEST = json.load(y_test_file)

	# Metrics calculation output
	output_file_path = os.path.join(DIR_PATH, "entropy.json")
	data = {}

	for file_name in FILE_NAMES:
		file_path = os.path.join(DIR_PATH, file_name)
		
		splitted_results = {}
		with open(file_path, 'r') as file:
			results = json.load(file)
			splitted_results["clean"] = split_correct(results["clean"], Y_TEST)
			splitted_results["adv"] = split_correct(results["adv"], Y_TEST)

		data[file_name] = {"clean": {}, "adv": {}}
		data[file_name]["clean"]["correct"] = get_shannon_entropy(splitted_results["clean"]["correct"])
		data[file_name]["clean"]["incorrect"] = get_shannon_entropy(splitted_results["clean"]["incorrect"])
		data[file_name]["adv"]["correct"] = get_shannon_entropy(splitted_results["adv"]["correct"])
		data[file_name]["adv"]["incorrect"] = get_shannon_entropy(splitted_results["adv"]["incorrect"])

	with open(output_file_path, 'w') as file:
		json.dump(data, file)


