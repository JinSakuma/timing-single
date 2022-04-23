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
from src.models.asr.rnnt.joint import RNNTransducer
from src.models.recognition.context_encoder import ContextEncoder
from src.models.recognition.tasks import (
    SystemActsPredictor,
    DialogActsPredictor,
)
from src.utils.utils import get_cer
torch.autograd.set_detect_anomaly(True)


class E2ESLU(nn.Module):

	def __init__(self, config, device, asr_input_dim, asr_num_class, dialog_acts_num_class):
		super().__init__()
		self.config = config
		self.device = device
		self.asr_input_dim = asr_input_dim
		self.asr_num_class = asr_num_class
		self.dialog_acts_num_class = dialog_acts_num_class
		self.create_models()

	def create_models(self):
		#self.config.loss_params.asr_weight>0.0:
		asr_model = RNNTransducer(
			device = self.device,
			num_classes=self.asr_num_class,
			input_dim=self.asr_input_dim,
			num_encoder_layers=self.config.model_params.num_encoder_layers,
			num_decoder_layers=self.config.model_params.num_decoder_layers,
			encoder_hidden_dim=self.config.model_params.encoder_hidden_dim,
			decoder_hidden_dim=self.config.model_params.decoder_hidden_dim,
			output_dim=self.config.model_params.output_dim,
			rnn_type=self.config.model_params.rnn_type,
			bidirectional=self.config.model_params.bidirectional,
		)
		self.asr_model = asr_model
		self.embedding_dim = 512 #asr_model.embedding_dim

		self.context_encoder = ContextEncoder(self.device)
		self.bert_hidden_dim = self.context_encoder.hidden_size

		dialog_acts_model = DialogActsPredictor(
			self.bert_hidden_dim+self.embedding_dim,
			self.dialog_acts_num_class,
			self.device,
		)
		self.dialog_acts_model = dialog_acts_model

		system_acts_model = SystemActsPredictor(
			self.bert_hidden_dim+self.embedding_dim,
			self.dialog_acts_num_class,
			self.device,
		)
		self.system_acts_model = system_acts_model

	def configure_optimizer_parameters(self):
		if  self.config.loss_params.asr_weight==1.0:
			parameters = chain(
				self.asr_model.parameters(),
			)
		elif self.config.loss_params.asr_weight==0.0:
			parameters = chain(
				self.context_encoder.parameters(),
				self.dialog_acts_model.parameters(),
				self.system_acts_model.parameters(),
			)
		else:
			parameters = chain(
				self.asr_model.parameters(),
				self.context_encoder.parameters(),
				self.dialog_acts_model.parameters(),
				self.system_acts_model.parameters(),
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
		inputs = batch[2].to(self.device)
		input_lengths = batch[3]#.to(self.device)
		labels = batch[4].to(self.device)
		label_lengths = batch[5]#.to(self.device)
		dialog_acts_labels = batch[7].to(self.device)
		#bert_labels = batch[10].to(self.device)
		#bert_masks = batch[11].to(self.device)
		roles = batch[12]#.to(self.device)
		batch_size = len(indices)
		
		inputs_no_pad = []
		labels_no_pad = []
		for i in range(batch_size):
			inputs_no_pad.append(inputs[i,:uttr_nums[i],:, :])
			labels_no_pad.append(labels[i,:uttr_nums[i],:])
		inputs = torch.cat(inputs_no_pad, dim=0)
		labels = torch.cat(labels_no_pad, dim=0)	
			
		input_lengths = input_lengths.cpu().view(-1)
		input_lengths = input_lengths[input_lengths>0]
		label_lengths = label_lengths.cpu().view(-1)
		label_lengths = label_lengths[label_lengths>0]
		
		log_probs, embedding = self.asr_model(inputs, input_lengths, labels, label_lengths)
		# print(log_probs.shape, input_lengths.shape, labels.shape, label_lengths.shape)
		asr_loss = self.get_asr_loss(log_probs, input_lengths, labels, label_lengths)
	
		if self.config.loss_params.dialog_acts_weight>0 or self.config.loss_params.system_acts_weight>0:
			# context encoding
			pooled_out = self.context_encoder(bert_labels, bert_masks, uttr_nums)
			b, d, h = pooled_out.shape
			embedding = embedding.view(b, d, -1)
			embedding = torch.cat([embedding, pooled_out], dim=-1)

			# Remove padding
			emb_no_pad = []
			roles_no_pad = []
			dialog_acts_no_pad = []
			for i in range(batch_size):
				emb_no_pad.append(embedding[i,:uttr_nums[i], :])
				roles_no_pad.append(roles[i,:uttr_nums[i]])
				dialog_acts_no_pad.append(dialog_acts_labels[i,:uttr_nums[i],:])
			embedding = torch.cat(emb_no_pad, dim=0)
			roles = torch.cat(roles_no_pad, dim=0)
			dialog_acts_labels = torch.cat(dialog_acts_no_pad, dim=0)	

		if self.config.loss_params.dialog_acts_weight>0:
			dialog_acts_probs = self.dialog_acts_model(embedding)
			dialog_acts_loss = self.dialog_acts_model.get_loss(dialog_acts_probs, dialog_acts_labels)
			dialog_acts_p, dialog_acts_r, dialog_acts_f1, dialog_acts_acc = self.f1_score_dialog_acts(dialog_acts_probs, dialog_acts_labels)
			num_dialog_acts_total = dialog_acts_probs.shape[0]
		else:
			dialog_acts_loss = 0
			dialog_acts_p, dialog_acts_r, dialog_acts_f1, dialog_acts_acc = 0, 0, 0, 0
			num_dialog_acts_total = 0

		if self.config.loss_params.system_acts_weight>0:
			system_acts_probs, system_acts_labels = self.system_acts_model(embedding, dialog_acts_labels, roles, uttr_nums)
			system_acts_loss = self.system_acts_model.get_loss(system_acts_probs, system_acts_labels)
			system_acts_p, system_acts_r, system_acts_f1, system_acts_acc = self.f1_score_dialog_acts(system_acts_probs, system_acts_labels)
			num_system_acts_total = system_acts_probs.shape[0]
		else:
			system_acts_loss = 0
			system_acts_p, system_acts_r, system_acts_f1, system_acts_acc = 0, 0, 0, 0
			num_system_acts_total = 0
		
		loss = (
			asr_loss * self.config.loss_params.asr_weight + 
			dialog_acts_loss * self.config.loss_params.dialog_acts_weight +
			system_acts_loss * self.config.loss_params.system_acts_weight
		)

		if split == "val":
			with torch.no_grad():
				hypotheses, hypothesis_lengths = self.asr_model.recognize(inputs, input_lengths)
	
			asr_cer = get_cer(hypotheses.cpu(), hypothesis_lengths.cpu(), labels.cpu(), label_lengths.cpu())
		else:
			asr_cer = 0
		outputs = {
			f'{split}_loss': loss,
			f'{split}_asr_loss': asr_loss,
			f'{split}_dialog_acts_loss': dialog_acts_loss,
			f'{split}_system_acts_loss': system_acts_loss,
			f'{split}_asr_cer': asr_cer,
			f'{split}_dialog_acts_precision': dialog_acts_p,
			f'{split}_dialog_acts_recall': dialog_acts_r,
			f'{split}_dialog_acts_f1': dialog_acts_f1,
			f'{split}_dialog_acts_acc': dialog_acts_acc,
			f'{split}_num_dialog_acts_total': num_dialog_acts_total,
			f'{split}_system_acts_precision': system_acts_p,
			f'{split}_system_acts_recall': system_acts_r,
			f'{split}_system_acts_f1': system_acts_f1,
			f'{split}_system_acts_acc': system_acts_acc,
			f'{split}_num_system_acts_total': num_system_acts_total,
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
		inputs = batch[2].to(self.device)
		input_lengths = batch[3].to(self.device)
		labels = batch[4].to(self.device)
		label_lengths = batch[5].to(self.device)
		dialog_acts_labels = batch[7].to(self.device)
		bert_labels = batch[10].to(self.device)
		bert_masks = batch[11].to(self.device)
		roles = batch[12].to(self.device)
		batch_size = len(indices)
		
		#if self.config.loss_params.asr_weight>0:
		log_probs, embedding, labels = self.asr_model(inputs, input_lengths, labels, uttr_nums)
			
		## Remove padding
		#probs_no_pad = []
		#labels_no_pad = []
		#for i in range(batch_size):
		#	probs_no_pad.append(log_probs[i,:uttr_nums[i],:, :])
		#	labels_no_pad.append(labels[i,:uttr_nums[i],:])
		#log_probs = torch.cat(probs_no_pad, dim=0)
		#labels = torch.cat(labels_no_pad, dim=0)

		#input_lengths = input_lengths.cpu().view(-1)
		#input_lengths = input_lengths[input_lengths>0]
		#label_lengths = label_lengths.cpu().view(-1)
		#label_lengths = label_lengths[label_lengths>0]
        
		hyp_list, hyplen_list, ref_list, reflen_list = [], [], [], []
		with torch.no_grad():
			for i in range(batch_size):
				hypothesis, hypothesis_lengths, references, reference_lengths = self.asr_model.decode(log_probs[i], input_lengths[i], labels[i], label_lengths[i])
				hyp_list.append(hypothesis)
				hyplen_list.append(hypothesis_lengths)
				ref_list.append(references)
				reflen_list.append(reference_lengths)
		
		return hyp_list, hyplen_list, ref_list, reflen_list