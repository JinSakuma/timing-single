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
from src.models.asr.distiller import CTC
#from src.models.recognition.context_encoder2 import ContextEncoder2
#from src.models.recognition.tasks import (
#    SystemActsPredictor,
#    DialogActsPredictor,
#)
from src.utils.utils import get_cer
torch.autograd.set_detect_anomaly(True)


class E2ESLU(nn.Module):

	def __init__(self, config, device, asr_input_dim, asr_num_class):
		super().__init__()
		self.config = config
		self.device = device
		self.asr_input_dim = asr_input_dim
		self.asr_num_class = asr_num_class
		#self.dialog_acts_num_class = dialog_acts_num_class
		self.create_models()

	def create_models(self):
		#self.config.loss_params.asr_weight>0.0:
		asr_model = CTC(
			self.device,
			self.asr_input_dim,
			self.asr_num_class,
			num_layers=self.config.model_params.num_layers,
			hidden_dim=self.config.model_params.hidden_dim,
			bidirectional=self.config.model_params.bidirectional,
		)
		self.asr_model = asr_model
		#self.embedding_dim = asr_model.embedding_dim

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
		indices = batch[1]
		wavs = batch[2]#.to(self.device)
		#input_lengths = batch[3].to(self.device)
		labels = batch[3].to(self.device)
		label_lengths = batch[4].to(self.device)
		batch_size = len(indices)
		
		inputs = []
		for i in range(batch_size):
			inputs += [w.to(self.device) for  w in wavs[i]]
		#if self.config.loss_params.asr_weight>0:
		log_probs, input_lengths, embedding = self.asr_model(inputs)
		
		# Remove padding
		#probs_no_pad = []
		labels_no_pad = []
		for i in range(batch_size):
		#	probs_no_pad.append(log_probs[i,:uttr_nums[i],:, :])
			labels_no_pad.append(labels[i,:uttr_nums[i],:])
		#log_probs = torch.cat(probs_no_pad, dim=0)
		labels = torch.cat(labels_no_pad, dim=0)	
		#	
		#input_lengths = input_lengths.cpu().view(-1)
		#input_lengths = input_lengths[input_lengths>0]
		label_lengths = label_lengths.cpu().view(-1)
		label_lengths = label_lengths[label_lengths>0]

		#print(log_probs.shape, input_lengths.shape, labels.shape, label_lengths.shape)
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
	
	def f1_score_dialog_acts(self, outputs, labels):
    
		P, R, F1, acc = 0, 0, 0, 0
		outputs = torch.sigmoid(outputs)

		for i in range(outputs.shape[0]):
			TP, FP, FN = 0, 0, 0
			for j in range(outputs.shape[1]):
				if outputs[i][j] > 0.5 and labels[i][j] == 1:
					TP += 1
				elif outputs[i][j] <= 0.5 and labels[i][j] == 1:
					FN += 1
				elif outputs[i][j] > 0.5 and labels[i][j] == 0:
					FP += 1
			precision = TP / float(TP + FP) if (TP + FP) != 0 else 0
			recall = TP / float(TP + FN) if (TP + FN) != 0 else 0
			F1 += 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
			P += precision
			R += recall

			p = (torch.where(outputs[i]>0.5)[0])
			r = (torch.where(labels[i]==1)[0])
			if len(p) == len(r) and (p == r).all():
				acc += 1

		P /= outputs.shape[0]
		R /= outputs.shape[0]
		F1 /= outputs.shape[0]
		return P, R, F1, acc	

	def test(self, batch):
		uttr_nums = batch[0]
		indices = batch[1]
		inputs = batch[2].to(self.device)
		input_lengths = batch[3].to(self.device)
		labels = batch[4].to(self.device)
		label_lengths = batch[5].to(self.device)
		dialog_acts_labels = batch[7].to(self.device)
		bert_labels = batch[10].to(self.device)
		bert_masks = batch[11].to(self.device)
		roles = batch[12].to(self.device)
		batch_size = len(indices)

		# context encoding
		pooled_out = self.context_encoder(bert_labels, bert_masks, uttr_nums)
		b, d, h = pooled_out.shape

		# asr for current utterance
		uttr_nums2 = batch[3][0].unsqueeze(0).to(self.device)
		inputs2 = batch[2][0][-1].unsqueeze(0).unsqueeze(0).to(self.device)
		input_lengths2 = batch[3][0][-1].unsqueeze(0).unsqueeze(0).to(self.device)
		labels2 = batch[4][0][-1].unsqueeze(0).unsqueeze(0).to(self.device)

		_, embedding, _ = self.asr_model(inputs2, input_lengths2, labels2, uttr_nums2)

		# concat and inference
		features = torch.cat([embedding, pooled_out[:, -1, :]], dim=-1)
		output = self.dialog_acts_model(features)
		
		return output

	def decode_asr(self, batch):
		uttr_nums = batch[0]
		indices = batch[1]
		inputs = batch[2]
		input_lengths = batch[3]
		labels = batch[4]
		label_lengths = batch[5]
		bert_label = batch[9]
		bert_masks = batch[10]
		batch_size = len(indices)
	
		# Remove padding
		inputs_no_pad = []
		labels_no_pad = []
		for i in range(batch_size):
			inputs_no_pad.append(inputs[i,:uttr_nums[i],:, :])
			labels_no_pad.append(labels[i,:uttr_nums[i],:])
		inputs = torch.cat(inputs_no_pad, dim=0)
		labels = torch.cat(labels_no_pad, dim=0)	
		
		log_probs, embedding, labels, input_lengths = self.asr_model(inputs, input_lengths, labels, uttr_nums)
		label_lengths = label_lengths.cpu().view(-1)
		label_lengths = label_lengths[label_lengths>0]
		hypotheses, hypothesis_length, references, reference_lengths = self.asr_model.decode(log_probs, input_lengths, labels, label_lengths)
		
		return hypotheses, hypothesis_length, references, reference_lengths
