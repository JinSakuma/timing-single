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

from src.models.timing.vad import VoiceActivityDetector
from src.models.asr.transformer.subsampling import Conv2dSubsampling5
torch.autograd.set_detect_anomaly(True)


class System(nn.Module):

	def __init__(self, config, device, asr_input_dim, asr_num_class):
		super().__init__()
		self.config = config
		self.device = device
		self.duration = 300 # 発話後3秒間を評価対象にしている
		self.asr_input_dim = asr_input_dim
		self.asr_num_class = asr_num_class
		self.create_models()

	def create_models(self):
		self.subsampling = Conv2dSubsampling5(self.asr_input_dim)
		vad = VoiceActivityDetector(
			self.device,
			self.config.model_params.input_dim,
			self.config.model_params.timing_hidden_dim,
		)
		self.vad = vad

	def configure_optimizer_parameters(self):

		parameters = chain(
			self.subsampling.parameters(),            
			self.vad.parameters(),
		)
		return parameters

	def forward(self, batch, split='train'):
		uttr_nums = batch[0]
		indices = batch[1]
		cnnae = batch[2].to(self.device)
		fbank = batch[3].to(self.device)
		input_lengths = batch[4]#.to(self.device)
		labels = batch[5].to(self.device)
		label_lengths = batch[6]#.to(self.device)
		uttr_labels = batch[7].to(self.device)
		batch_size = int(uttr_nums)


		vad_loss, vad_acc = 0, 0
		subsampled = self.subsampling(fbank)        
		outputs = self.vad(subsampled, input_lengths)
		for i in range(uttr_nums):
			output = outputs[i]
			vad_loss = vad_loss+self.vad.get_loss(output[:input_lengths[i]], uttr_labels[i][:input_lengths[i]])
			vad_acc = vad_acc+self.vad.get_acc(output[:input_lengths[i]], uttr_labels[i][:input_lengths[i]])
		vad_acc = vad_acc / float(uttr_nums)


		outputs = {
			f'{split}_loss': vad_loss,
			f'{split}_vad_acc': vad_acc,
		}
		
		return outputs
	
