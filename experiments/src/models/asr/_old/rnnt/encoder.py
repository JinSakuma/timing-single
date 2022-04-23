import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

from src.utils.utils import edit_distance


class Encoder(nn.Module):
    """
    Encoder of RNN-Transducer.
    Args:
        input_dim (int): dimension of input vector
        hidden_state_dim (int, optional): hidden state dimension of encoder (default: 320)
        output_dim (int, optional): output dimension of encoder and decoder (default: 512)
        num_layers (int, optional): number of encoder layers (default: 4)
        rnn_type (str, optional): type of rnn cell (default: lstm)
        bidirectional (bool, optional): if True, becomes a bidirectional encoder (default: True)
    Inputs: inputs, input_lengths
        inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
            `FloatTensor` of size ``(batch, seq_length, dimension)``.
        input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
    Returns:
        (Tensor, Tensor)
        * outputs (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
            ``(batch, seq_length, dimension)``
        * hidden_states (torch.FloatTensor): A hidden state of encoder. `FloatTensor` of size
            ``(batch, seq_length, dimension)``
    """
    supported_rnns = {
        'lstm': nn.LSTM,
        'gru': nn.GRU,
        'rnn': nn.RNN,
    }

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int,
            rnn_type: str = 'lstm',
            dropout_p: float = 0.2,
            bidirectional: bool = True,
    ):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        rnn_cell = self.supported_rnns[rnn_type.lower()]
        self.rnn = rnn_cell(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=dropout_p,
            bidirectional=bidirectional,
        )
        self.out_proj = nn.Linear(hidden_dim << 1 if bidirectional else hidden_dim, output_dim)

    def count_parameters(self) -> int:
        """ Count parameters of encoder """
        return sum([p.numel for p in self.parameters()])

    def update_dropout(self, dropout_p: float) -> None:
        """ Update dropout probability of encoder """
        for name, child in self.named_children():
            if isinstance(child, nn.Dropout):
                child.p = dropout_p

    def forward(self, inputs, input_lengths):
        """
        Forward propagate a `inputs` for  encoder training.
        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
        Returns:
            (Tensor, Tensor)
            * outputs (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            * output_lengths (torch.LongTensor): The length of output tensor. ``(batch)``
        """
        t = inputs.size(1)        
        inputs = rnn_utils.pack_padded_sequence(
            inputs, 
            input_lengths, 
            batch_first=True,
            enforce_sorted=False,
        )
        
        outputs, (rnn_h, rnn_c) = self.rnn(inputs)
        outputs, _ = rnn_utils.pad_packed_sequence(
            outputs, 
            batch_first=True,
            padding_value=0.,
            total_length=t,
        )

        outputs = self.out_proj(outputs)  # (b*d, t, l)
        #embedding = self.combine_h_and_c(rnn_h, rnn_c)

        return outputs #, embedding


    def combine_h_and_c(self, h, c):
        batch_size = h.size(1)
        h = h.permute(1, 0, 2).contiguous()
        c = c.permute(1, 0, 2).contiguous()
        h = h.view(batch_size, -1)
        c = c.view(batch_size, -1)
        
        return torch.cat([h, c], dim=1)

