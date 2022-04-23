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

from src.models.asr.rnn.ctc import CTC
from src.models.timing.rtg import RTG
from src.models.recognition.context_encoder import ContextEncoder
from src.models.recognition.tasks2 import (
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
		self.duration = 300 # 発話後3秒間を評価対象にしている
		self.asr_input_dim = asr_input_dim
		self.asr_num_class = asr_num_class
		self.dialog_acts_num_class = dialog_acts_num_class
		self.create_models()

	def create_models(self):
		#self.config.loss_params.asr_weight>0.0:
		asr_model = CTC(
			input_dim=self.asr_input_dim,
			num_class=self.asr_num_class,
			num_layers=self.config.model_params.num_layers,
			hidden_dim=self.config.model_params.asr_hidden_dim,
			bidirectional=self.config.model_params.bidirectional,
		)
		self.asr_model = asr_model
		self.embedding_dim = asr_model.embedding_dim

		self.context_encoder = ContextEncoder(self.device)
		self.bert_hidden_dim = self.context_encoder.hidden_size

		timing_model = RTG(
			self.device,
			#self.config.model_params.timing_hidden_dim+self.config.model_params.hidden_dim*2+self.config.data_params.n_mels,
			self.config.model_params.asr_hidden_dim+self.config.model_params.hidden_dim*2+self.config.data_params.n_mels,
			self.config.model_params.timing_hidden_dim,
		)
		self.timing_model = timing_model
	
		dialog_acts_model = DialogActsPredictor(
			#self.bert_hidden_dim+self.embedding_dim,
			self.config.model_params.asr_hidden_dim+self.config.data_params.n_mels+self.bert_hidden_dim,
			self.config.model_params.hidden_dim,
			self.dialog_acts_num_class,
			self.device,
		)
		self.dialog_acts_model = dialog_acts_model

		system_acts_model = SystemActsPredictor(
			#self.bert_hidden_dim+self.embedding_dim,
			self.config.model_params.asr_hidden_dim+self.config.data_params.n_mels+self.bert_hidden_dim,
			self.config.model_params.hidden_dim,
			self.dialog_acts_num_class,
			self.device,
		)
		self.system_acts_model = system_acts_model

	def configure_optimizer_parameters(self):
		if  self.config.loss_params.asr_weight==1.0:
			parameters = chain(
				self.asr_model.parameters(),
			)
		elif  self.config.loss_params.timing_weight==1.0:
			parameters = chain(
				self.timing_model.parameters(),
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
		uttr_type = batch[1]
		inputs = batch[2].to(self.device)
		input_lengths = batch[3]#.to(self.device)
		timings = batch[4].to(self.device)
		uttr_labels = batch[5].to(self.device)
		labels = batch[6].to(self.device)
		label_lengths = batch[7]#.to(self.device)
		dialog_acts_labels = batch[8].to(self.device)
		system_acts_labels = batch[9].to(self.device)
		bert_labels = batch[11].to(self.device)
		bert_masks = batch[12].to(self.device)
		offset = batch[13]
		batch_size = inputs.size(0)
		
		inputs_no_pad = []
		timings_no_pad = []
		uttr_labels_no_pad = []
		labels_no_pad = []
		for i in range(batch_size):
			inputs_no_pad.append(inputs[i,:uttr_nums[i],:, :])
			timings_no_pad.append(timings[i,:uttr_nums[i],:])
			uttr_labels_no_pad.append(uttr_labels[i,:uttr_nums[i],:])
			labels_no_pad.append(labels[i,:uttr_nums[i],:])
		inputs = torch.cat(inputs_no_pad, dim=0)
		timings = torch.cat(timings_no_pad, dim=0)	
		uttr_labels = torch.cat(uttr_labels_no_pad, dim=0)	
		labels = torch.cat(labels_no_pad, dim=0)	
			
		input_lengths = input_lengths.cpu().view(-1)
		input_lengths = input_lengths[input_lengths>0]
		label_lengths = label_lengths.cpu().view(-1)
		label_lengths = label_lengths[label_lengths>0]
		
		log_probs, embedding, logits = self.asr_model(inputs, input_lengths)
		#print(inputs.shape, logits.shape, input_lengths.shape, labels.shape, label_lengths.shape)
		if self.config.loss_params.asr_weight>0:
			asr_loss = self.get_asr_loss(log_probs, input_lengths, labels, label_lengths)
		else:
			asr_loss = 0

		embedding = torch.cat([inputs, logits], dim=-1)
	
		if self.config.loss_params.dialog_acts_weight>-1 or self.config.loss_params.system_acts_weight>-1:
			t = embedding.size(1)
			# context encoding
			pooled_out = self.context_encoder(bert_labels, bert_masks, uttr_nums)
			pooled_out = pooled_out.unsqueeze(2).repeat(1,1,t,1)
			b, d, t, h = pooled_out.shape
			pooled_out = pooled_out.view(b*d, t, -1)
			emb = torch.cat([embedding, pooled_out], dim=-1)

			# Remove padding
			dialog_acts_no_pad = []
			system_acts_no_pad = []
			for i in range(batch_size):
				dialog_acts_no_pad.append(dialog_acts_labels[i,:uttr_nums[i],:])
				system_acts_no_pad.append(system_acts_labels[i,:uttr_nums[i],:])
			dialog_acts_labels = torch.cat(dialog_acts_no_pad, dim=0)	
			system_acts_labels = torch.cat(system_acts_no_pad, dim=0)	

		if self.config.loss_params.dialog_acts_weight>-1:
			dialog_acts_probs, dialog_acts_emb = self.dialog_acts_model(emb, input_lengths)

			prob_list = []
			for i in range(d):
				prob_list.append(dialog_acts_probs[i, input_lengths[i]-1, :].unsqueeze(0))
			dialog_acts_probs = torch.cat(prob_list, dim=0)

			dialog_acts_loss = self.dialog_acts_model.get_loss(dialog_acts_probs, dialog_acts_labels)
			dialog_acts_p, dialog_acts_r, dialog_acts_f1, dialog_acts_acc = self.f1_score_dialog_acts(dialog_acts_probs, dialog_acts_labels)
			num_dialog_acts_total = dialog_acts_probs.shape[0]
		else:
			dialog_acts_loss = 0
			dialog_acts_p, dialog_acts_r, dialog_acts_f1, dialog_acts_acc = 0, 0, 0, 0
			num_dialog_acts_total = 0

		if self.config.loss_params.system_acts_weight>-1:
			#roles = [int(i) // 2 for i in uttr_type[0]]
			system_acts_probs, system_acts_labels, system_acts_emb = self.system_acts_model(emb, input_lengths, system_acts_labels, uttr_type[0], uttr_nums)
			prob_list = []
			label_list = []
			for i in range(d):
				if uttr_type[0][i]==0:
					prob_list.append(system_acts_probs[i, input_lengths[i]-1, :].unsqueeze(0))
					label_list.append(system_acts_labels[i].unsqueeze(0))
			if len(prob_list)==0:
				print(uttr_type)
			system_acts_probs = torch.cat(prob_list, dim=0)
			system_acts_labels = torch.cat(label_list, dim=0)
			system_acts_loss = self.system_acts_model.get_loss(system_acts_probs, system_acts_labels)
			system_acts_p, system_acts_r, system_acts_f1, system_acts_acc = self.f1_score_dialog_acts(system_acts_probs, system_acts_labels)
			num_system_acts_total = system_acts_probs.shape[0]
		else:
			system_acts_loss = 0
			system_acts_p, system_acts_r, system_acts_f1, system_acts_acc = 0, 0, 0, 0
			num_system_acts_total = 0

		timing_loss = 0
		if self.config.loss_params.timing_weight>0:
			assert batch_size==1, "batch size must be set 1"
			for i in range(uttr_nums[0]):
				if uttr_type[0][i] == 0 or uttr_type[0][i] == 1:
					if offset[0][i]>3000:
						continue
					context = torch.cat([dialog_acts_emb[i], system_acts_emb[i]], dim=-1)
					emb = torch.cat([embedding[i], context], dim=-1)
					emb = emb[:input_lengths[i]]

					output = self.timing_model(emb, uttr_labels[i][:input_lengths[i]])
					if uttr_type[0][i]==0:
						timing_loss = timing_loss+self.timing_model.get_loss(output[-self.duration:], timings[i][:input_lengths[i]][-self.duration:])	
					elif uttr_type[0][i]==1:
						timing_loss = timing_loss+self.timing_model.get_loss(output[-offset[0][i]:], timings[i][:input_lengths[i]][-offset[0][i]:])	
					

		loss = (
			asr_loss * self.config.loss_params.asr_weight + 
			timing_loss * self.config.loss_params.timing_weight +
			dialog_acts_loss * self.config.loss_params.dialog_acts_weight +
			system_acts_loss * self.config.loss_params.system_acts_weight
		)

		if split == "val": #and self.config.loss_params.asr_weight>0:
			with torch.no_grad():
				hypotheses, hypothesis_lengths, references, reference_lengths = \
                			self.asr_model.decode(
                    		log_probs, input_lengths,
                    		labels, label_lengths
				)
	
			asr_cer = get_cer(hypotheses, hypothesis_lengths, labels, label_lengths)
		else:
			asr_cer = 0
		
		outputs = {
			f'{split}_loss': loss,
			f'{split}_asr_loss': asr_loss,
			f'{split}_timing_loss': dialog_acts_loss,
			f'{split}_dialog_acts_loss': dialog_acts_loss,
			f'{split}_system_acts_loss': system_acts_loss,
			f'{split}_asr_cer': asr_cer,
			#f'{split}_timing_precision': dialog_acts_p,
			#f'{split}_timing_recall': dialog_acts_r,
			#f'{split}_timing_f1': dialog_acts_f1,
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
