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

from src.models.encoders.bert_encoder import BertEncoder

from src.models.asr.hubert.hubert import CTC
from src.models.dialog_acts.transformer import (
    SystemActsPredictor,
    DialogActsPredictor,
)
from src.utils.utils import get_cer
torch.autograd.set_detect_anomaly(True)

from transformers import BertTokenizer, BertModel, BertForMaskedLM
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

with open('src/datasets/vocab/subwords.txt') as f:
    subwords_list = f.read().split("\n")
    
def id2token(id_list):
    idx = [int(subwords_list[i]) for i in id_list]
    return tokenizer.convert_ids_to_tokens(idx)

def id2bertid(id_list):
    idx = [subwords_list[i] for i in id_list]
    return idx


def id2token2(id_list):  
    return tokenizer.convert_ids_to_tokens(id_list)


class SLU(nn.Module):

    def __init__(self, config, device, asr_input_dim, asr_num_class, dialog_acts_num_class, system_acts_num_class, encoding_dim=256):
        super().__init__()
        self.config = config
        self.device = device
        self.duration = 300 # 発話後3秒間を評価対象にしている
        self.asr_input_dim = 768 #asr_input_dim
        self.asr_num_class = asr_num_class
        self.dialog_acts_num_class = dialog_acts_num_class
        self.system_acts_num_class = system_acts_num_class
        self.encoding_dim = encoding_dim
        self.create_models()

    def create_models(self):
        #self.config.loss_params.asr_weight>0.0:
        asr_model = CTC(
            self.device,
            input_dim=self.asr_input_dim,
            num_class=self.asr_num_class,
            num_layers=self.config.model_params.num_layers,
            bidirectional=self.config.model_params.bidirectional,
        )
        self.asr_model = asr_model
        self.embedding_dim = asr_model.embedding_dim
        
        self.bert_encoder = BertEncoder(self.device, self.encoding_dim)
        self.bert_hidden_dim = self.bert_encoder.hidden_size

        dialog_acts_model = DialogActsPredictor(
            self.embedding_dim+self.encoding_dim,
            self.dialog_acts_num_class,
            self.device,
        )
        self.dialog_acts_model = dialog_acts_model

        system_acts_model = SystemActsPredictor(
            self.embedding_dim+self.encoding_dim,
            self.system_acts_num_class,
            self.device,
        )
        self.system_acts_model = system_acts_model
        
    def freeze(self, model):
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
            else:
                print(name)

    def configure_optimizer_parameters(self):
        if  self.config.loss_params.asr_weight==1.0:
            parameters = chain(
                self.asr_model.parameters(),
            )
        elif self.config.loss_params.asr_weight==0.0:
#             self.freeze(self.dialog_acts_model)
#             self.freeze(self.system_acts_model)
#             print('freezed')
            parameters = chain(
                # self.bert_encoder.parameters(),
                self.dialog_acts_model.parameters(),
                self.system_acts_model.parameters(),
            )
        else:
            parameters = chain(
                self.asr_model.parameters(),
                # self.bert_encoder.parameters(),
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

    def pad_wav(self, wav, maxlen=2000, pad=0):
        b, length, dim = wav.shape
        padded = torch.zeros((b, maxlen, dim)) + pad
        padded.to(self.device)

        assert length<=maxlen, 'length too large'
        padded[:, :length, :] = wav

        return padded

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
        
        asr_loss = 0
        bert_labels = []
        bert_masks = []
        log_probs_list, input_lengths_list, embedding_list = [], [], []
#         for i in range(batch_size):

        i=0  # batch_size=1
        inputs = [w[:int(duration[i][j]*16*50)].to(self.device) for j, w in enumerate(wavs[i])]
        for j in range(uttr_nums[i]):
            log_prob, length, emb = self.asr_model([inputs[j]])
            log_probs_list.append(log_prob)
            input_lengths_list.append(length)
            embedding_list.append(emb)

        max_len = max(input_lengths_list)
        log_probs_list = [self.pad_wav(log_prob, max_len) for log_prob in log_probs_list]
        embedding_list = [self.pad_wav(emb, max_len) for emb in embedding_list]

        log_probs = torch.cat(log_probs_list, dim=0)
        input_lengths = torch.tensor(input_lengths_list)
        embedding = torch.cat(embedding_list, dim=0).to(self.device)
        
        if self.config.loss_params.asr_weight>0:
            asr_loss = asr_loss + self.get_asr_loss(log_probs, input_lengths, labels[i], label_lengths[i])
        else:
            asr_loss = 0

        asr_cer = 0
        total = 1
#         with torch.no_grad():
#             if split=='val':
#                 hypotheses, hypothesis_lengths, references, reference_lengths = \
#                     self.asr_model.decode(
#                     log_probs, input_lengths,
#                     labels[i], label_lengths[i]
#                 )
#                 asr_cer += get_cer(hypotheses, hypothesis_lengths, references, reference_lengths)
#                 total += 1
        
        hypotheses, hypothesis_lengths, references, reference_lengths = \
                self.asr_model.decode(
                log_probs, input_lengths,
                labels[i], label_lengths[i]
            )
        
        transcripts = []
        for j in range(len(hypotheses)):
            #if uttr_type[0][i]==2 or uttr_type[0][i]==3:
            hypo = hypotheses[j]
            transcript = ' '.join(id2token(hypo)).replace('[CLS] ', '').replace(' [SEP]', '')
            transcripts.append(transcript)
        #    else:
        #        ref = references[i]
        #        transcript = ' '.join(id2token(ref)).replace('[CLS] ', '').replace(' [SEP]', '')
        #        transcripts.append(transcript)


        max_length = 70
        result = tokenizer(transcripts, max_length=max_length, padding="max_length", truncation=True, return_tensors='pt')
        labels = result['input_ids']
        masks = result['attention_mask']

        bert_labels.append(labels.unsqueeze(0))
        bert_masks.append(masks.unsqueeze(0))
        bert_labels = torch.cat(bert_labels, dim=0).to(self.device)
        bert_masks = torch.cat(bert_masks, dim=0).to(self.device)

        t = embedding.size(1)
        # context encoding
        pooled_out = self.bert_encoder(bert_labels, bert_masks, uttr_nums)
        pooled_out_ = pooled_out.unsqueeze(0).unsqueeze(2).repeat(1,1,t,1)
        b, d, t, h = pooled_out_.shape
        pooled_out_ = pooled_out_.view(b*d, t, -1)
        emb = torch.cat([embedding, pooled_out_], dim=-1)

        if self.config.loss_params.dialog_acts_weight>0:
            dialog_acts_probs, dialog_acts_emb = self.dialog_acts_model(emb, input_lengths)

            dialog_acts_loss = self.dialog_acts_model.get_loss(dialog_acts_probs, dialog_acts_labels[i])
            dialog_acts_p, dialog_acts_r, dialog_acts_f1, dialog_acts_acc = self.f1_score_dialog_acts(dialog_acts_probs, dialog_acts_labels[i])
            num_dialog_acts_total = dialog_acts_probs.shape[i]
        else:
            dialog_acts_loss = 0
            dialog_acts_p, dialog_acts_r, dialog_acts_f1, dialog_acts_acc = 0, 0, 0, 0
            num_dialog_acts_total = 0

        if self.config.loss_params.system_acts_weight>0:
            system_acts_probs, system_acts_emb = self.system_acts_model(emb, input_lengths)
            system_acts_loss = self.system_acts_model.get_loss(system_acts_probs, system_acts_labels[i])
            system_acts_p, system_acts_r, system_acts_f1, system_acts_acc = self.f1_score_dialog_acts(system_acts_probs, system_acts_labels[i])
            num_system_acts_total = system_acts_probs.shape[i]
        else:
            system_acts_loss = 0
            system_acts_p, system_acts_r, system_acts_f1, system_acts_acc = 0, 0, 0, 0
            num_system_acts_total = 0

        loss = (
            #asr_loss * self.config.loss_params.asr_weight + 
            dialog_acts_loss * self.config.loss_params.dialog_acts_weight +
            system_acts_loss * self.config.loss_params.system_acts_weight
        )

        if split == "val": #and self.config.loss_params.asr_weight>0:
        	asr_cer = asr_cer / float(total)
        else:
        	asr_cer = 0

        outputs = {
            f'{split}_loss': loss,
            f'{split}_asr_loss': asr_loss,
            f'{split}_timing_loss': dialog_acts_loss,
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
            'dialog_acts_probs': dialog_acts_probs,
            'system_acts_probs': system_acts_probs,
            'dialog_acts_emb': dialog_acts_emb,
            'system_acts_emb': system_acts_emb,
            'bert_encoding': pooled_out,
        }

        return outputs

    def recog(self, batch, split='train'):
        
        with torch.no_grad():
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

            asr_loss = 0
            bert_labels = []
            bert_masks = []
            log_probs_list, input_lengths_list, embedding_list = [], [], []

            i=0  # batch_size=1
            inputs = [w[:int(duration[i][j]*16*50)].to(self.device) for j, w in enumerate(wavs[i])]
            for j in range(uttr_nums[i]):
                log_prob, length, emb = self.asr_model([inputs[j]])
                log_probs_list.append(log_prob)
                input_lengths_list.append(length)
                embedding_list.append(emb)

            max_len = max(input_lengths_list)
            log_probs_list = [self.pad_wav(log_prob, max_len) for log_prob in log_probs_list]
            embedding_list = [self.pad_wav(emb, max_len) for emb in embedding_list]

            log_probs = torch.cat(log_probs_list, dim=0)
            input_lengths = torch.tensor(input_lengths_list)
            embedding = torch.cat(embedding_list, dim=0).to(self.device)

#             if self.config.loss_params.asr_weight>0:
#                 asr_loss = asr_loss + self.get_asr_loss(log_probs, input_lengths, labels[i], label_lengths[i])
#             else:
#                 asr_loss = 0

            asr_cer = 0
            total = 1
    #         with torch.no_grad():
    #             if split=='val':
    #                 hypotheses, hypothesis_lengths, references, reference_lengths = \
    #                     self.asr_model.decode(
    #                     log_probs, input_lengths,
    #                     labels[i], label_lengths[i]
    #                 )
    #                 asr_cer += get_cer(hypotheses, hypothesis_lengths, references, reference_lengths)
    #                 total += 1

            hypotheses, hypothesis_lengths, references, reference_lengths = \
                    self.asr_model.decode(
                    log_probs, input_lengths,
                    labels[i], label_lengths[i]
                )

            transcripts = []
            for j in range(len(hypotheses)):
                #if uttr_type[0][i]==2 or uttr_type[0][i]==3:
                hypo = hypotheses[j]
                transcript = ' '.join(id2token(hypo)).replace('[CLS] ', '').replace(' [SEP]', '')
                transcripts.append(transcript)
            #    else:
            #        ref = references[i]
            #        transcript = ' '.join(id2token(ref)).replace('[CLS] ', '').replace(' [SEP]', '')
            #        transcripts.append(transcript)


            max_length = 70
            result = tokenizer(transcripts, max_length=max_length, padding="max_length", truncation=True, return_tensors='pt')
            labels = result['input_ids']
            masks = result['attention_mask']

            bert_labels.append(labels.unsqueeze(0))
            bert_masks.append(masks.unsqueeze(0))
            bert_labels = torch.cat(bert_labels, dim=0).to(self.device)
            bert_masks = torch.cat(bert_masks, dim=0).to(self.device)

            t = embedding.size(1)
            # context encoding
            pooled_out = self.bert_encoder(bert_labels, bert_masks, uttr_nums)
            pooled_out_ = pooled_out.unsqueeze(0).unsqueeze(2).repeat(1,1,t,1)
            b, d, t, h = pooled_out_.shape
            pooled_out_ = pooled_out_.view(b*d, t, -1)
            emb = torch.cat([embedding, pooled_out_], dim=-1)

#             if self.config.loss_params.dialog_acts_weight>0:
            dialog_acts_probs, dialog_acts_emb = self.dialog_acts_model(emb, input_lengths)
#                 dialog_acts_loss = self.dialog_acts_model.get_loss(dialog_acts_probs, dialog_acts_labels[0])
#                 dialog_acts_p, dialog_acts_r, dialog_acts_f1, dialog_acts_acc = self.f1_score_dialog_acts(dialog_acts_probs, dialog_acts_labels[0])
#                 num_dialog_acts_total = dialog_acts_probs.shape[0]
#             else:
#                 dialog_acts_loss = 0
#                 dialog_acts_p, dialog_acts_r, dialog_acts_f1, dialog_acts_acc = 0, 0, 0, 0
#                 num_dialog_acts_total = 0

#             if self.config.loss_params.system_acts_weight>0:
            system_acts_probs, system_acts_emb = self.system_acts_model(emb, input_lengths)
            #prob_list = []
            #label_list = []
            #for i in range(len(uttr_type[0])):
            #    if uttr_type[0][i]==0:
            #        prob_list.append(system_acts_probs[i].unsqueeze(0))
            #        label_list.append(system_acts_labels[0][i].unsqueeze(0))
            #if len(prob_list)>0:
            #    system_acts_probs = torch.cat(prob_list, dim=0)
            #    system_acts_labels = torch.cat(label_list, dim=0)
      
            #else:
            #    system_acts_loss = 0
            #    system_acts_p, system_acts_r, system_acts_f1, system_acts_acc = 0, 0, 0, 0
            #    num_system_acts_total = 0
      
                    
#                 if len(prob_list)>0:
#                     system_acts_probs = torch.cat(prob_list, dim=0)
#                     system_acts_labels = torch.cat(label_list, dim=0)
#                     system_acts_loss = self.system_acts_model.get_loss(system_acts_probs, system_acts_labels)
#                     system_acts_p, system_acts_r, system_acts_f1, system_acts_acc = self.f1_score_dialog_acts(system_acts_probs, system_acts_labels)
#                     num_system_acts_total = system_acts_probs.shape[0]
#                 else:
#                     system_acts_loss = 0
#                     system_acts_p, system_acts_r, system_acts_f1, system_acts_acc = 0, 0, 0, 0
#                     num_system_acts_total = 0
#             else:
#                 system_acts_loss = 0
#                 system_acts_p, system_acts_r, system_acts_f1, system_acts_acc = 0, 0, 0, 0
#                 num_system_acts_total = 0

#             loss = (
#                 asr_loss * self.config.loss_params.asr_weight + 
#                 dialog_acts_loss * self.config.loss_params.dialog_acts_weight +
#                 system_acts_loss * self.config.loss_params.system_acts_weight
#             )

#             if split == "val": #and self.config.loss_params.asr_weight>0:
#                 asr_cer = asr_cer / float(total)
            #    with torch.no_grad():
            #        hypotheses, hypothesis_lengths, references, reference_lengths = \
            #                    self.asr_model.decode(
            #                    log_probs, input_lengths,
            #                    labels, label_lengths
            #        )

            #    asr_cer = get_cer(hypotheses, hypothesis_lengths, labels, label_lengths)
            #else:
            #    asr_cer = 0

            outputs = {
                'dialog_acts_emb': dialog_acts_emb,
                'system_acts_emb': system_acts_emb,
                'dialog_acts_probs': dialog_acts_probs,
                'system_acts_probs': system_acts_probs,
                'bert_encoding': pooled_out,
#                 f'{split}_loss': loss,
#                 f'{split}_asr_loss': asr_loss,
#                 f'{split}_timing_loss': dialog_acts_loss,
#                 f'{split}_dialog_acts_loss': dialog_acts_loss,
#                 f'{split}_system_acts_loss': system_acts_loss,
#                 f'{split}_asr_cer': asr_cer,
#                 #f'{split}_timing_precision': dialog_acts_p,
#                 #f'{split}_timing_recall': dialog_acts_r,
#                 #f'{split}_timing_f1': dialog_acts_f1,
#                 f'{split}_dialog_acts_precision': dialog_acts_p,
#                 f'{split}_dialog_acts_recall': dialog_acts_r,
#                 f'{split}_dialog_acts_f1': dialog_acts_f1,
#                 f'{split}_dialog_acts_acc': dialog_acts_acc,
#                 f'{split}_num_dialog_acts_total': num_dialog_acts_total,
#                 f'{split}_system_acts_precision': system_acts_p,
#                 f'{split}_system_acts_recall': system_acts_r,
#                 f'{split}_system_acts_f1': system_acts_f1,
#                 f'{split}_system_acts_acc': system_acts_acc,
#                 f'{split}_num_system_acts_total': num_system_acts_total,
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
