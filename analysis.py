import json
import os
import numpy as np

from scipy.special import softmax

def logits_to_softmax(logits):
	np_logits = np.array(logits)
	np_softmax = softmax(np_logits, axis=1)
	# np_softmax[np_softmax==0] = 1
	return np_softmax

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

def calc_metric_by_class(func, file_name, results_softmax, targets):
	data = {}
	data[file_name] = {"clean": {}, "adv": {}}
	data[file_name]["clean"]["correct"] = func(results_softmax["clean"]["correct"], targets)
	data[file_name]["clean"]["incorrect"] = func(results_softmax["clean"]["incorrect"], targets)
	data[file_name]["adv"]["correct"] = func(results_softmax["adv"]["correct"], targets)
	data[file_name]["adv"]["incorrect"] = func(results_softmax["adv"]["incorrect"], targets)
	return data

# Max softmax prob
def get_max_softmax(preds, targets):
	return list(preds.max(axis=1))

# Shannon's entropy
def get_shannon_entropy(preds, targets):
	preds[preds==0] = 1
	entropy = -np.sum(preds * np.log2(preds), axis=1)
	return list(entropy)

def get_brier(preds, targets, **args):
    one_hot_targets = np.zeros(preds.shape)
    one_hot_targets[np.arange(len(targets)), targets] = 1.0
    return list(np.mean(np.sum((preds - one_hot_targets) ** 2, axis=1)))

def get_ll(preds, targets, **args):
    return list(np.log(1e-12 + preds[np.arange(len(targets)), targets]).mean())

# Expected calibration error (binning assigned prob scores and compared against average accuracies within bins)
def get_ece(preds, targets, n_bins=15, **args):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    confidences, predictions = np.max(preds, 1), np.argmax(preds, 1)
    accuracies = np.equal(predictions, targets).astype(int)
    
    ece = 0.0
    avg_confs_in_bins = []
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            delta = avg_confidence_in_bin - accuracy_in_bin
            avg_confs_in_bins.append(delta)
            ece += np.abs(delta) * prop_in_bin
        else:
            avg_confs_in_bins.append(None)
    # For reliability diagrams, also need to return these:
    # return ece, bin_lowers, avg_confs_in_bins
    return ece

def get_tace(preds, targets, n_bins=15, threshold=1e-3, **args):
    n_objects, n_classes = preds.shape
    
    res = 0.0
    for cur_class in range(n_classes):
        cur_class_conf = preds[:, cur_class]
        
        targets_sorted = targets[cur_class_conf.argsort()]
        cur_class_conf_sorted = np.sort(cur_class_conf)
        
        targets_sorted = targets_sorted[cur_class_conf_sorted > threshold]
        cur_class_conf_sorted = cur_class_conf_sorted[cur_class_conf_sorted > threshold]
        
        bin_size = len(cur_class_conf_sorted) // n_bins
                
        for bin_i in range(n_bins):
            bin_start_ind = bin_i * bin_size
            if bin_i < n_bins-1:
                bin_end_ind = bin_start_ind + bin_size
            else:
                bin_end_ind = len(targets_sorted)
                bin_size = bin_end_ind - bin_start_ind  # extend last bin until the end of prediction array
            bin_acc = (targets_sorted[bin_start_ind : bin_end_ind] == cur_class)
            bin_conf = cur_class_conf_sorted[bin_start_ind : bin_end_ind]
            avg_confidence_in_bin = np.mean(bin_conf)
            avg_accuracy_in_bin = np.mean(bin_acc)
            delta = np.abs(avg_confidence_in_bin - avg_accuracy_in_bin)
#             print(f'bin size {bin_size}, bin conf {avg_confidence_in_bin}, bin acc {avg_accuracy_in_bin}')
            res += delta * bin_size / (n_objects * n_classes)
            
    return res

if __name__ == "__main__":

	DIR_PATH = os.getcwd()
	FILE_NAMES = ["no-at-single-logits.json", "no-at-ensemble-logits.json", "PGD-single-logits.json", "PGD-ensemble-logits.json"]

	with open(os.path.join(DIR_PATH, "y_test.json"), 'r') as y_test_file:
		Y_TEST = json.load(y_test_file)

	# Metrics calculation output
	# Metrics used: entropy, ece, 
	output_file_paths = ["entropy.json", "ece.json", "max_softmax.json", "brier.json", "ll.json"]
	# output_file_paths = ["ece.json"]
	for output_file in output_file_paths:
		print(output_file)
		data = {}

		for file_name in FILE_NAMES:
			file_path = os.path.join(DIR_PATH, file_name)
			
			splitted_results = {}
			with open(file_path, 'r') as file:
				logits = json.load(file)
				splitted_results["clean"] = split_correct(logits["clean"], Y_TEST)
				splitted_results["adv"] = split_correct(logits["adv"], Y_TEST)

			results_softmax = {"clean": {}, "adv": {}}
			results_softmax["clean"]["correct"] = logits_to_softmax(splitted_results["clean"]["correct"])
			results_softmax["clean"]["incorrect"] = logits_to_softmax(splitted_results["clean"]["incorrect"])
			results_softmax["adv"]["correct"] = logits_to_softmax(splitted_results["adv"]["correct"])
			results_softmax["adv"]["incorrect"] = logits_to_softmax(splitted_results["adv"]["incorrect"])

			if output_file == "entropy.json":
				data = calc_metric_by_class(get_shannon_entropy, file_name, results_softmax, Y_TEST)

			if output_file == "ece.json":
				data[file_name] = {}
				all_clean = np.concatenate([results_softmax['clean']['correct'], results_softmax['clean']['incorrect']])
				all_adv = np.concatenate([results_softmax['adv']['correct'], results_softmax['adv']['incorrect']])
				data[file_name]['clean'] = get_ece(all_clean, Y_TEST)
				data[file_name]['adv'] = get_ece(all_adv, Y_TEST)

			# if output_file == "tace.json":
			# 	data[file_name] = {}
			# 	all_clean = np.concatenate([results_softmax['clean']['correct'], results_softmax['clean']['incorrect']])
			# 	all_adv = np.concatenate([results_softmax['adv']['correct'], results_softmax['adv']['incorrect']])
			# 	data[file_name]['clean'] = get_tace(all_clean, Y_TEST)
			# 	data[file_name]['adv'] = get_tace(all_adv, Y_TEST)

			if output_file == "max_softmax.json":
				data = calc_metric_by_class(get_max_softmax, file_name, results_softmax, Y_TEST)

			if output_file == "brier.json":
				data = calc_metric_by_class(get_brier, file_name, results_softmax, Y_TEST)

			if output_file == "ll.json":
				data = calc_metric_by_class(get_ll, file_name, results_softmax, Y_TEST)

		output_file_path = os.path.join(DIR_PATH, output_file)
		with open(output_file_path, 'w') as file:
			json.dump(data, file)


