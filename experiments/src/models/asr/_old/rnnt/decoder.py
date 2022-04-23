import torch.nn as nn
from torch import Tensor
from typing import Tuple


class Decoder(nn.Module):
    """
    Decoder of RNN-Transducer
    Args:
        num_classes (int): number of classification
        hidden_state_dim (int, optional): hidden state dimension of decoder (default: 512)
        output_dim (int, optional): output dimension of encoder and decoder (default: 512)
        num_layers (int, optional): number of decoder layers (default: 1)
        rnn_type (str, optional): type of rnn cell (default: lstm)
        sos_id (int, optional): start of sentence identification
        eos_id (int, optional): end of sentence identification
        dropout_p (float, optional): dropout probability of decoder
    Inputs: inputs, input_lengths
        inputs (torch.LongTensor): A target sequence passed to decoder. `IntTensor` of size ``(batch, seq_length)``
        input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
        hidden_states (torch.FloatTensor): A previous hidden state of decoder. `FloatTensor` of size
            ``(batch, seq_length, dimension)``
    Returns:
        (Tensor, Tensor):
        * decoder_outputs (torch.FloatTensor): A output sequence of decoder. `FloatTensor` of size
            ``(batch, seq_length, dimension)``
        * hidden_states (torch.FloatTensor): A hidden state of decoder. `FloatTensor` of size
            ``(batch, seq_length, dimension)``
    """
    supported_rnns = {
        'lstm': nn.LSTM,
        'gru': nn.GRU,
        'rnn': nn.RNN,
    }

    def __init__(
            self,
            num_classes: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int,
            rnn_type: str = 'lstm',
            sos_id: int = 1,
            eos_id: int = 2,
            dropout_p: float = 0.2,
    ):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.embedding = nn.Embedding(num_classes, hidden_dim)
        rnn_cell = self.supported_rnns[rnn_type.lower()]
        self.rnn = rnn_cell(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=dropout_p,
            bidirectional=False,
        )
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def count_parameters(self) -> int:
        """ Count parameters of encoder """
        return sum([p.numel for p in self.parameters()])

    def update_dropout(self, dropout_p: float) -> None:
        """ Update dropout probability of encoder """
        for name, child in self.named_children():
            if isinstance(child, nn.Dropout):
                child.p = dropout_p
        
    def forward(
            self,
            inputs: Tensor,
            input_lengths: Tensor = None,
            hidden_states: Tensor = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward propage a `inputs` (targets) for training.
        Args:
            inputs (torch.LongTensor): A target sequence passed to decoder. `IntTensor` of size ``(batch, seq_length)``
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
            hidden_states (torch.FloatTensor): A previous hidden state of decoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
        Returns:
            (Tensor, Tensor):
            * decoder_outputs (torch.FloatTensor): A output sequence of decoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            * hidden_states (torch.FloatTensor): A hidden state of decoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
        """
        t = inputs.size(1)
        embedded = self.embedding(inputs)

        if input_lengths is not None:          
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded,
                input_lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            outputs, hidden_states = self.rnn(embedded, hidden_states)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(
                outputs,
                batch_first=True,
                padding_value=0.,
                total_length=t,
            )
            outputs = self.out_proj(outputs)
        else:
            outputs, hidden_states = self.rnn(embedded, hidden_states)
            outputs = self.out_proj(outputs)

        return outputs, hidden_states

