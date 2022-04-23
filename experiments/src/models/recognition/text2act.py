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
from src.datasets.my_harper_valley2 import MyHarperValley2, create_dataloader
from src.models.recognition.asr import CTC
from src.models.recognition.context_encoder2 import ContextEncoder2
from src.models.recognition.tasks import (
    TaskTypePredictor,
    DialogActsPredictor,
    SentimentPredictor,
)
from src.utils.utils import get_cer
torch.autograd.set_detect_anomaly(True)


class E2ESLU(nn.Module):

	def __init__(self, config, device, dialog_acts_num_class):
		super().__init__()
		self.config = config
		self.device = device
		self.dialog_acts_num_class = dialog_acts_num_class
		self.create_models()

	def create_models(self):

		self.context_encoder = ContextEncoder2(self.device)
		self.bert_hidden_dim = self.context_encoder.hidden_size

		dialog_acts_model = DialogActsPredictor(
			self.bert_hidden_dim,
			self.dialog_acts_num_class,
			self.device,
		)
		self.dialog_acts_model = dialog_acts_model

	def configure_optimizer_parameters(self):
		parameters = chain(
			self.context_encoder.parameters(),
			self.dialog_acts_model.parameters(),
		)

		return parameters

	def forward(self, batch, split='train'):
		uttr_nums = batch[0]
		indices = batch[1]
		inputs = batch[2].to(self.device)
		input_lengths = batch[3].to(self.device)
		labels = batch[4].to(self.device)
		label_lengths = batch[5].to(self.device)
		dialog_acts_labels = batch[7].to(self.device)
		bert_labels = batch[10].to(self.device)
		bert_masks = batch[11].to(self.device)
		batch_size = len(indices)
		
		if self.config.loss_params.dialog_acts_weight>0:
			# context encoding
			embedding = self.context_encoder(bert_labels, bert_masks, uttr_nums)
			#b, d, h = pooled_out.shape
			#embedding = embedding.view(b, d, -1)
			#embedding = torch.cat([embedding, pooled_out], dim=-1)

			# Remove padding
			emb_no_pad = []
			dialog_acts_no_pad = []
			for i in range(batch_size):
				emb_no_pad.append(embedding[i,:uttr_nums[i], :])
				dialog_acts_no_pad.append(dialog_acts_labels[i,:uttr_nums[i],:])
			embedding = torch.cat(emb_no_pad, dim=0)
			dialog_acts_labels = torch.cat(dialog_acts_no_pad, dim=0)	

			dialog_acts_probs = self.dialog_acts_model(embedding)
			dialog_acts_loss = self.dialog_acts_model.get_loss(dialog_acts_probs, dialog_acts_labels)
			precision, recall, f1, acc = self.f1_score_dialog_acts(dialog_acts_probs, dialog_acts_labels)
			num_dialog_acts_total = dialog_acts_probs.shape[0]
		else:
			dialog_acts_loss = 0
			precision, recall, f1, acc = 0, 0, 0, 0
			num_dialog_acts_total = 0

		
		loss = (
			asr_loss * self.config.loss_params.asr_weight + 
			dialog_acts_loss * self.config.loss_params.dialog_acts_weight
		)

		outputs = {
			f'{split}_loss': loss,
			f'{split}_asr_loss': asr_loss,
			f'{split}_dialog_acts_loss': dialog_acts_loss,
			f'{split}_asr_cer': asr_cer,
			f'{split}_dialog_acts_precision': precision,
			f'{split}_dialog_acts_recall': recall,
			f'{split}_dialog_acts_f1': f1,
			f'{split}_dialog_acts_acc': acc,
			f'{split}_num_dialog_acts_total': num_dialog_acts_total,
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
