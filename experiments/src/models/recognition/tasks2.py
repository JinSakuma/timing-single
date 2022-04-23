import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils


class TaskTypePredictor(nn.Module):

    def __init__(self, input_dim, task_type_num_classes):
        super().__init__()
        self.task_fc = nn.Linear(input_dim, task_type_num_classes)
        self.task_type_num_classes = task_type_num_classes

    def forward(self, inputs):
        task_logits = self.task_fc(inputs)
        task_log_probs = F.log_softmax(task_logits, dim=1)
        return task_log_probs
    
    def get_loss(self, log_probs, targets):
        return F.nll_loss(log_probs, targets)


class DialogActsPredictor(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_dialog_acts, device):
        super().__init__()

        self.lstm = torch.nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                batch_first=True,
            )

        self.dialogacts_fc = nn.Linear(hidden_dim, num_dialog_acts)
        self.num_dialog_acts = num_dialog_acts
        self.criterion = nn.BCEWithLogitsLoss(reduction='sum').to(device)

    def forward(self, inputs, input_lengths):
        t = inputs.size(1)
       
        inputs = rnn_utils.pack_padded_sequence(
            inputs, 
            input_lengths, 
            batch_first=True,
            enforce_sorted=False,
        )

        # outputs : batch_size x maxlen x hidden_dim
        # rnn_h   : num_layers * num_directions, batch_size, hidden_dim
        # rnn_c   : num_layers * num_directions, batch_size, hidden_dim
        outputs, _ = self.lstm(inputs)
        outputs, _ = rnn_utils.pad_packed_sequence(
            outputs, 
            batch_first=True,
            padding_value=0.,
            total_length=t,
        )
        dialogacts_logits = self.dialogacts_fc(outputs)
        # one person can have multiple dialog actions
        # dialogacts_probs = torch.sigmoid(dialogacts_logits)
        return dialogacts_logits, outputs

    def get_loss(self, probs, targets):
        # probs   : batch_size x num_dialog_acts
        # targets : batch_size x num_dialog_acts
        return self.criterion(probs, targets.float())


class SystemActsPredictor(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_dialog_acts, device):
        super().__init__()
        
        self.lstm = torch.nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                batch_first=True,
            )
        self.system_acts_fc = nn.Linear(hidden_dim, num_dialog_acts)
        self.num_dialog_acts = num_dialog_acts
        self.criterion = nn.BCEWithLogitsLoss(reduction='sum').to(device)

    def forward(self, inputs, input_lengths, labels, roles, uttr_nums):
        
        head = []
        tmp = 0 
        for i in range(len(uttr_nums)):
            head.append(tmp+uttr_nums[i])
            tmp = uttr_nums[i]

        #inputs_list = []
        #length_list = []
        #labels_list = []
        #for i, role in enumerate(roles):
        #    if role==0 and i>0 and i not in head:
        #        inputs_list.append(inputs[i].unsqueeze(0))
        #        length_list.append(input_lengths[i].unsqueeze(0))
        #        labels_list.append(labels[i].unsqueeze(0))
        #inputs = torch.cat(inputs_list, dim=0)
        #input_lengths = torch.cat(length_list, dim=0)
        #labels = torch.cat(labels_list, dim=0)

        t = inputs.size(1)
       
        inputs = rnn_utils.pack_padded_sequence(
            inputs, 
            input_lengths, 
            batch_first=True,
            enforce_sorted=False,
        )

        # outputs : batch_size x maxlen x hidden_dim
        # rnn_h   : num_layers * num_directions, batch_size, hidden_dim
        # rnn_c   : num_layers * num_directions, batch_size, hidden_dim
        outputs, _ = self.lstm(inputs)
        outputs, _ = rnn_utils.pad_packed_sequence(
            outputs, 
            batch_first=True,
            padding_value=0.,
            total_length=t,
        )

        system_acts_logits = self.system_acts_fc(outputs)
        # one person can have multiple dialog actions
        # dialogacts_probs = torch.sigmoid(dialogacts_logits)
        return system_acts_logits, labels, outputs

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

