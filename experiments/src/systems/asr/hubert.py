import os
import sys
import numpy as np
from tqdm import tqdm
from itertools import chain
from collections import OrderedDict
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.utils.utils import (
    AverageMeter,
    save_checkpoint as save_snapshot,
    copy_checkpoint as copy_snapshot,
)
from src.models.asr.hubert.hubert import CTC
from src.utils.utils import get_cer
torch.autograd.set_detect_anomaly(True)


class System(nn.Module):

	def __init__(self, config, device, asr_num_class):
		super().__init__()
		self.config = config
		self.device = device
		self.asr_input_dim = self.config.model_params.asr_input_dim
		self.asr_num_class = asr_num_class
		#self.dialog_acts_num_class = dialog_acts_num_class
		self.create_models()

	def create_models(self):
		asr_model = CTC(
			self.device,
			self.asr_input_dim,
			self.asr_num_class,
			num_layers=self.config.model_params.num_layers,
			bidirectional=self.config.model_params.bidirectional,
		)
		self.asr_model = asr_model

	def configure_optimizer_parameters(self):
		if  self.config.loss_params.asr_weight>0.0:
			parameters = chain(
				self.asr_model.parameters(),
			)

		return parameters

	def get_asr_loss(self, log_probs, input_lengths, labels, label_lengths):
		loss = self.asr_model.get_loss(
			log_probs,
			input_lengths,
			labels,
			label_lengths,
			blank=0,
		)
		return loss

	def forward(self, batch, split='train'):
		uttr_nums = batch[0]
		wavs = batch[2]#.to(self.device)
		labels = batch[6].to(self.device)
		label_lengths = batch[7].to(self.device)
		batch_size = len(wavs)        

		inputs = [w.to(self.device) for  w in wavs]
		log_probs, input_lengths, embedding = self.asr_model(inputs)
        
		asr_loss = self.get_asr_loss(log_probs, input_lengths, labels, label_lengths)
		loss = asr_loss * self.config.loss_params.asr_weight 

		with torch.no_grad():
			if split=="val":
			#if self.config.loss_params.asr_weight>0:
				if isinstance(log_probs, tuple):
					log_probs = log_probs[1]

				hypotheses, hypothesis_lengths, references, reference_lengths = \
					self.asr_model.decode(
						log_probs, input_lengths,
						labels, label_lengths,
#self.tr#ain_dataset.sos_index,
#self.tr#ain_dataset.eos_index,
#self.tr#ain_dataset.pad_index,
#self.tr#ain_dataset.eps_index,
						)	
				asr_cer = get_cer(hypotheses, hypothesis_lengths, references, reference_lengths)
			else:
				asr_cer = 0

		outputs = {
			f'{split}_loss': loss,
			f'{split}_asr_loss': asr_loss,
			f'{split}_asr_cer': asr_cer,
		}
		
		return outputs
	