import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
import json
import os

from adv.utils import classify, get_acc, get_logger
from adv.dataset_utils import load_mnist_all
from adv.mnist_model import BasicModel, BatchNormModel, EnsembleModel
from adv.pgd_attack import PGDAttack

def load_ensemble_model(name):
    path = os.path.join('saved_models', name)
    model_files = os.listdir(path)
    single_models = []
    for file in model_files:
        net = BatchNormModel()
        net = net.eval().to(device)
        net.load_state_dict(torch.load(os.path.join(path, file)))
        single_models.append(net)
    return EnsembleModel(single_models).to(device)

def classify_dump_ensemble(ensemble, x, batch_size=200, num_classes=10, method='aggregate_vote'):
    with torch.no_grad():
        y_preds = []
        for net in ensemble.models:
            y_pred = torch.zeros((x.size(0), num_classes)).to('cuda')
            for i in range(int(np.ceil(x.size(0) / batch_size))):
                begin = i * batch_size
                end = (i + 1) * batch_size
                y_pred[begin:end] = net(x[begin:end].to('cuda'))
            y_preds.append(y_pred.tolist())
    return y_preds

def load_validation(model_name, ensemble=False):

    with open('test_mnist.yml', 'r') as stream:
        config = yaml.safe_load(stream)

    # Set up device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set up num_classes
    num_classes = 10

    # Set all random seeds
    seed = config['meta']['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set up model directory
    save_dir = os.path.join(config['meta']['save_path'], 'saved_models')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)

    # Load model
    if ensemble:
    	log = get_logger('dump_validation', 'ensemble')
    else:
    	log = get_logger('dump_validation', model_name)
	log.info('Loading model...')
	if ensemble:
		net = load_ensemble_model('mnist_ensemble')
	else:
	    net = BatchNormModel()
	    net = net.eval().to(device)

	    # have to handle model loaded from CAT18 differently
	    if 'cat18' in model_path:
	        net = torch.nn.DataParallel(net)
	        net.load_state_dict(torch.load(model_path + '.pt')['net'])
	        net = net.module
	    else:
	        net.load_state_dict(torch.load(model_path + '.pt'))

	    if device == 'cuda':
	        if len(config['meta']['gpu_id']) > 1:
	            net = torch.nn.DataParallel(net)
	            net = net.eval()
	        cudnn.benchmark = True

    # Load dataset
    # Validation inputs
    log.info('Preparing data...')
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_mnist_all(
    data_dir=config['meta']['data_path'], val_size=0.1, shuffle=True,
    seed=seed)
    batch_size = config['test']['batch_size']

    # Generate validation outputs
    # for batch_idx, (inputs, targets) in enumerate(validloader):
    #     inputs, targets = inputs.to(device), targets.to(device)
    #     y_preds = classify(net, inputs, num_classes=num_classes)

    # num_valid_samples = x_valid.shape[0]
    if ensemble:
    	y_pred = classify_dump_ensemble(net, x_valid, num_classes=num_classes)
    else:
	    if config['pgd']['quant']:
	        y_pred = classify(net, quantize(
	            x_valid), num_classes=num_classes)
	    else:
	        y_pred = classify(
	            net, x_valid, num_classes=num_classes)

    valid_output_path = "./valid_output/" + model_name + ".json"
    data = { "clean": y_pred.tolist() }
    # acc = get_acc(y_pred, y_valid)
    # log.info('Clean acc: %.4f', acc)

    # Start PGD attack
    log.info('Starting PGD attack...')
    attack = PGDAttack(net, x_train, y_train)
    x_adv = attack(x_valid, y_valid, batch_size=batch_size, **config['pgd'])
    if ensemble:
    	y_pred_adv = classify_dump_ensemble(net, x_adv, num_classes=num_classes)
    else:
    	y_pred_adv = classify(net, x_adv, num_classes=num_classes)

    data["adv"] = y_pred_adv.tolist()
    with open(valid_output_path, 'w') as file:
        json.dump(data, file)
    # acc = get_acc(y_pred_adv, y_valid)
    # log.info('Adv acc: %.4f', acc)


if __name__ == "__main__":
	load_validation("mnist_ensemble", ensemble=True)

