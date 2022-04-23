import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

from src.models.asr.transformer.encoder import Encoder
from src.models.asr.transformer.nets_utils import get_subsample
from src.models.asr.transformer.nets_utils import make_non_pad_mask

idim=256
adim=256
odim=256
transformer_encpder_selfattn_layer_type="selfattn"
attention_dim=256
aheads=4
wshare=4
conv_kernel_length="21_23_25_27_29_31_33_35_37_39_41_43"
conv_usebias=False
eunits=1024
elayers=3
transformer_input_layer=None #"conv2d"
dropout_rate=0.1
positional_dropout_rate=0.1
attention_dropout_rate=0.0
stochastic_depth_rate=0.0
intermediate_ctc_layers=""
ctc_softmax=None
conditioning_layer_dim=odim


class DialogActsPredictor(nn.Module):

    def __init__(self, input_dim, num_dialog_acts, device):
        super().__init__()

        self.encoder = Encoder(
            idim=idim,
            selfattention_layer_type=transformer_encpder_selfattn_layer_type,
            attention_dim=adim,
            attention_heads=aheads,
            conv_wshare=wshare,
            conv_kernel_length=conv_kernel_length,
            conv_usebias=conv_usebias,
            linear_units=eunits,
            num_blocks=elayers,
            input_layer=transformer_input_layer,
            dropout_rate=dropout_rate,
            positional_dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            stochastic_depth_rate=stochastic_depth_rate,
            intermediate_layers=None,
            ctc_softmax=None,
            conditioning_layer_dim=odim,
        )        

        self.fc1 = nn.Linear(input_dim, idim)
        self.fc2 = nn.Linear(odim, odim//4)
        self.fc3 = nn.Linear(odim//4, num_dialog_acts)
        self.num_dialog_acts = num_dialog_acts
        self.input_dim = input_dim
        self.criterion = nn.BCEWithLogitsLoss(reduction='sum').to(device)

    def forward(self, inputs, input_lengths):
        inputs = self.fc1(inputs)
        mask = make_non_pad_mask(input_lengths.tolist()).to(inputs.device).unsqueeze(-2)
        out, _ = self.encoder(inputs, mask)
        out = self.fc2(out.mean(1))
        dialogacts_logits = self.fc3(out)
        # one person can have multiple dialog actions
        # dialogacts_probs = torch.sigmoid(dialogacts_logits)
        return dialogacts_logits, out

    def get_loss(self, probs, targets):
        # probs   : batch_size x num_dialog_acts
        # targets : batch_size x num_dialog_acts
        return self.criterion(probs, targets.float())


class SystemActsPredictor(nn.Module):

    def __init__(self, input_dim, num_dialog_acts, device):
        super().__init__()

        self.encoder = Encoder(
            idim=idim,
            selfattention_layer_type=transformer_encpder_selfattn_layer_type,
            attention_dim=adim,
            attention_heads=aheads,
            conv_wshare=wshare,
            conv_kernel_length=conv_kernel_length,
            conv_usebias=conv_usebias,
            linear_units=eunits,
            num_blocks=elayers,
            input_layer=transformer_input_layer,
            dropout_rate=dropout_rate,
            positional_dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            stochastic_depth_rate=stochastic_depth_rate,
            intermediate_layers=None,
            ctc_softmax=None,
            conditioning_layer_dim=odim,
        )        

        self.fc1 = nn.Linear(input_dim, idim)
        self.fc2 = nn.Linear(odim, odim//4)
        self.fc3 = nn.Linear(odim//4, num_dialog_acts)
        
        self.num_dialog_acts = num_dialog_acts
        self.criterion = nn.BCEWithLogitsLoss(reduction='sum').to(device)

    def forward(self, inputs, input_lengths):
        
        inputs = self.fc1(inputs)
        mask = make_non_pad_mask(input_lengths.tolist()).to(inputs.device).unsqueeze(-2)
        out, _ = self.encoder(inputs, mask)
        out = self.fc2(out.mean(1))
        systemacts_logits = self.fc3(out)

        return systemacts_logits, out

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

