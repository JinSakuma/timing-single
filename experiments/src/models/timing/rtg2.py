import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SilenceEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=300):
        super(SilenceEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)#.transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, src, silence):
        tmp = torch.zeros(1, src.size(1), src.size(2))
        for i, s in enumerate(silence): 
            if s>0:
                if s >= self.max_len:
                    tmp[:, i, :] = self.pe[:, int(self.max_len-1):int(self.max_len), :]
                else:
                    tmp[:, i, :] = self.pe[:, int(s):int(s+1), :]

        device = src.device
        src = src + tmp.to(device)
        return self.dropout(src)


class RTG(nn.Module):

    def __init__(self, device, input_dim, hidden_dim, silence_encoding_type="silence_encoding"):
        super().__init__()
        
        self.device = device
        self.silence_encoding_type = silence_encoding_type
        if silence_encoding_type=="concat":
            input_dim += 1
        elif silence_encoding_type=="silence_encoding":
            self.silence_encoding = SilenceEncoding(input_dim)

        self.lstm = torch.nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                batch_first=True,
            )

        self.fc = nn.Linear(hidden_dim, 1)
        self.criterion = nn.BCEWithLogitsLoss(reduction='sum').to(device)

    def get_silence(self, uttr_label):
        silence = torch.zeros(len(uttr_label)).to(self.device)
        silence[0] = uttr_label[0]
        for i, u in enumerate(uttr_label):
            if u == 1:
                silence[i] = 0
            else:
                silence[i] = silence[i-1]+1
        return silence.unsqueeze(1)

    def forward(self, inputs, uttr_labels):
        silences = self.get_silence(uttr_labels)
        if self.silence_encoding_type=="concat":
            inputs = torch.cat([inputs, silences], dim=-1)
            inputs = inputs.unsqueeze(0)
        elif self.silence_encoding_type=="silence_encoding":
            inputs = inputs.unsqueeze(0)
            inputs = self.silence_encoding(inputs, silences)
        else:
            raise Exception('Not implemented')

        h, _ = self.lstm(inputs, None)
        logits = self.fc(h)
        logits = logits.view(-1)
        return logits
   
    def get_loss(self, probs, targets):
        return self.criterion(probs, targets.float())


class DialogActsPredictor(nn.Module):

    def __init__(self, input_dim, num_dialog_acts, device):
        super().__init__()
        self.dialogacts_fc = nn.Linear(input_dim, num_dialog_acts)
        self.num_dialog_acts = num_dialog_acts
        self.criterion = nn.BCEWithLogitsLoss(reduction='sum').to(device)

    def forward(self, inputs):
        dialogacts_logits = self.dialogacts_fc(inputs)
        # one person can have multiple dialog actions
        # dialogacts_probs = torch.sigmoid(dialogacts_logits)
        return dialogacts_logits

    def get_loss(self, probs, targets):
        # probs   : batch_size x num_dialog_acts
        # targets : batch_size x num_dialog_acts
        return self.criterion(probs, targets.float())


class SystemActsPredictor(nn.Module):

    def __init__(self, input_dim, num_dialog_acts, device):
        super().__init__()
        self.system_acts_fc = nn.Linear(input_dim, num_dialog_acts)
        self.num_dialog_acts = num_dialog_acts
        self.criterion = nn.BCEWithLogitsLoss(reduction='sum').to(device)

    def forward(self, inputs, labels, roles, uttr_nums):
        
        head = []
        tmp = 0 
        for i in range(len(uttr_nums)):
            head.append(tmp+uttr_nums[i])
            tmp = uttr_nums[i]

        inputs_list = []
        labels_list = []
        for i, role in enumerate(roles):
            if role==1 and i>0 and i not in head:
                inputs_list.append(inputs[i-1].unsqueeze(0))
                labels_list.append(labels[i].unsqueeze(0))
        inputs = torch.cat(inputs_list, dim=0)
        labels = torch.cat(labels_list, dim=0)

        system_acts_logits = self.system_acts_fc(inputs)
        # one person can have multiple dialog actions
        # dialogacts_probs = torch.sigmoid(dialogacts_logits)
        return system_acts_logits, labels

    def get_loss(self, probs, targets):
        # probs   : batch_size x num_dialog_acts
        # targets : batch_size x num_dialog_acts
        return self.criterion(probs, targets.float())


class SentimentPredictor(nn.Module):

    def __init__(self, input_dim, sentiment_num_classes):
        super().__init__()
        self.sentiment_fc = nn.Linear(input_dim, sentiment_num_classes)
        self.sentiment_num_classes = sentiment_num_classes

    def forward(self, inputs):
        sentiment_logits = self.sentiment_fc(inputs)
        sentiment_log_probs = F.log_softmax(sentiment_logits, dim=1)
        return sentiment_log_probs

    def get_loss(self, pred_log_probs, target_probs):
        # pred_logits   : batch_size x num_sentiment_class
        # target_logits : batch_size x num_sentiment_class
        xentropy = -torch.sum(target_probs * pred_log_probs, dim=1)
        return torch.mean(xentropy)


class SpeakerIdPredictor(nn.Module):

    def __init__(self, input_dim, num_speaker_ids):
        super().__init__()
        self.speaker_id_fc = nn.Linear(input_dim, num_speaker_ids)
        self.num_speaker_ids = num_speaker_ids

    def forward(self, inputs):
        speaker_id_logits = self.speaker_id_fc(inputs)
        speaker_id_log_probs = F.log_softmax(speaker_id_logits, dim=1)
        return speaker_id_log_probs

    def get_loss(self, log_probs, targets):
        return F.nll_loss(log_probs, targets)

