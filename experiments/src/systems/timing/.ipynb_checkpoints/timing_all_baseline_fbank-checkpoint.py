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
from src.models.asr.transformer.subsampling import Conv2dSubsampling5
from src.models.timing.rtg import RTG
from src.utils.utils import get_cer
torch.autograd.set_detect_anomaly(True)


class System(nn.Module):

	def __init__(self, config, device, asr_input_dim, asr_num_class, dialog_acts_num_class, system_acts_num_class):
		super().__init__()
		self.config = config
		self.device = device
		self.duration = 300 # 発話後3秒間を評価対象にしている
		self.asr_input_dim = asr_input_dim
		self.asr_num_class = asr_num_class
		self.dialog_acts_num_class = dialog_acts_num_class
		self.system_acts_num_class = system_acts_num_class
		self.create_models()
		self.T = 0 # config.model_params.pred_offset

	def create_models(self):
		self.subsampling = Conv2dSubsampling5(self.asr_input_dim)
		timing_model = RTG(
			self.device,
			self.config.model_params.input_dim+1, #+self.dialog_acts_num_class*2,
			self.config.model_params.timing_hidden_dim,
		)
		self.timing_model = timing_model
		

	def configure_optimizer_parameters(self):
		parameters = chain(
			self.subsampling.parameters(),
			self.timing_model.parameters(),
		)

		return parameters

	def forward(self, batch, split='train'):
		uttr_nums = batch[0]
		uttr_type = batch[1]
		wavs = batch[2]
		cnnae = batch[3].to(self.device)
		fbank = batch[4].to(self.device)
		input_lengths = batch[5]
		timings = batch[6].to(self.device)
		uttr_labels = batch[7].to(self.device)
		labels = batch[8].to(self.device)
		label_lengths = batch[9].to(self.device)
		dialog_acts_labels = batch[10].to(self.device)
		system_acts_labels = batch[11].to(self.device)
		offset = batch[12]
		duration = batch[13]
		batch_size = len(wavs)

		i=0
		subsampled = self.subsampling(fbank[i])
		length = max(input_lengths[i])
		embedding = subsampled[:, :length, :]
# 		embedding = torch.cat([subsampled[:, :length, :], cnnae[i][:, :length, :]], dim=-1)        
            
		del wavs

		asr_loss = 0

		timing_loss = 0
		if self.config.loss_params.timing_weight>0:
			assert batch_size==1, "batch size must be set 1"
			for j in range(uttr_nums[i]):
				#if offset[0][i]>300 or uttr_type[0][i] == 3:
				if uttr_type[i][j] == 1:
					continue

				t = embedding[j].size(0)
				if uttr_type[i][j] == 0 or uttr_type[i][j] == 1:
					speaker = torch.zeros([t, 1]).to(self.device)
				else:
					speaker = torch.ones([t, 1]).to(self.device)
					
				emb = torch.cat([embedding[j], speaker], dim=-1)
				dur = duration[i][j]

				emb = emb[:input_lengths[i][j]]
				output = self.timing_model(emb, uttr_labels[i][j][:input_lengths[i][j]])
				timing_loss = timing_loss+self.timing_model.get_loss(output[dur:], timings[i][j][:input_lengths[i][j]][dur:])
					
		loss = (
			asr_loss * self.config.loss_params.asr_weight + 
			timing_loss * self.config.loss_params.timing_weight
		)

		if split == "val" and self.config.loss_params.asr_weight>0:
			with torch.no_grad():
				hypotheses, hypothesis_lengths, references, reference_lengths = \
                			self.asr_model.decode(
                    		log_probs, input_lengths[i],
                    		labels[i], label_lengths[i]
				)
	
			asr_cer = get_cer(hypotheses, hypothesis_lengths, references, reference_lengths)
		else:
			asr_cer = 0
		
		outputs = {
			f'{split}_loss': loss,
			f'{split}_asr_loss': asr_loss,
			f'{split}_timing_loss': 0,
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