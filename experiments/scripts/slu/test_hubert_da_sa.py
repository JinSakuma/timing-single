import os
import torch
import numpy
import random
import getpass
from copy import deepcopy
#from src.systems.slu.slu3 import SLU
from src.systems.slu.slu_bert_wo_acoustic2 import SLU
from src.utils.setup import process_config
from src.utils.utils import load_json
from src.utils.tester_da_sa import tester
from src.datasets.timing_dataset2 import get_dataloader, get_dataset
import wandb


def run(config_path, gpu_device=-1):
	config = process_config(config_path)
	if gpu_device >= 0: config.gpu_device = gpu_device
	seed_everything(config.seed)
	ModelClass = globals()[config.system]
	if config.cuda:
		device = torch.device("cuda:{}".format(config.gpu_device))
	else:
		device = torch.device("cpu")

	if config.is_use_wandb:
		wandb.init(
		project='hvb_speech_system_action_subword', 
		dir='wandb',
		entity=getpass.getuser(), 
		name=config.exp_name, 
		config=config, 
		sync_tensorboard=True,
	)

	asr_target_type="subword"
	val_dataset = get_dataset(config, "val", asr_target_type=asr_target_type)
	test_dataset = get_dataset(config, "test", asr_target_type=asr_target_type)
	val_loader = get_dataloader(val_dataset, config, "val")
	test_loader = get_dataloader(test_dataset, config, "test")

	loader_dict = {"val": val_loader, "test": test_loader}
	#loader_dict = {"train": train_loader, "val": val_loader, "test": test_loader}

	model = ModelClass(config, device, config.model_params.hubert_input_dim, config.model_params.hubert_asr_num_class, val_dataset.dialog_acts_num_class, val_dataset.next_acts_num_class)
	del val_dataset
	del test_dataset
	#model.asr_model.load_state_dict(torch.load(config.asr_continue_from_checkpoint), strict=False)
	model.load_state_dict(torch.load(config.slu_continue_from_checkpoint), strict=False)
	#model.dialog_acts_model.load_state_dict(torch.load(config.da_continue_from_checkpoint), strict=False)
	#model.system_acts_model.load_state_dict(torch.load(config.sa_continue_from_checkpoint), strict=False)
	model.to(device)
	parameters = model.configure_optimizer_parameters()
	optimizer = torch.optim.AdamW(
		parameters,
		lr=config.optim_params.learning_rate,
		weight_decay=config.optim_params.weight_decay,
	)    

	tester(
		model=model,
		loader_dict=loader_dict,
		optimizer=optimizer,
		device=device,
		outdir=config.exp_dir,
		is_use_wandb=config.is_use_wandb,
	)


def seed_everything(seed):
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	numpy.random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('config', type=str, default='path to config file')
	parser.add_argument('--gpu-device', type=int, default=-1)
	args = parser.parse_args()
	run(args.config, gpu_device=args.gpu_device)
