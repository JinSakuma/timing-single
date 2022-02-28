import torch
import torch.nn as nn
import torch.nn.functional as F


class VoiceActivityDetector(nn.Module):

    def __init__(self, device, input_dim, hidden_dim): #, silence_encoding_type="concat"):
        super().__init__()
        
        self.device = device
        self.lstm = torch.nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                batch_first=True,
            )

        self.fc = nn.Linear(hidden_dim, 1)
        self.criterion = nn.BCEWithLogitsLoss(reduction='sum').to(device)

    def forward(self, inputs):
        inputs = inputs.unsqueeze(0)
        h, _ = self.lstm(inputs, None)
        logits = self.fc(h)
        logits = logits.view(logits.size(1))
        return logits

    def recog(self, inputs, input_lengths):
        outs = []
        with torch.no_grad():
            for i in range(len(input_lengths)):
                output = self.forward(inputs[i][:input_lengths[i]])
                outs.append(torch.sigmoid(output))            
        return outs

    def get_loss(self, probs, targets):
        return self.criterion(probs, targets.float())

    def get_acc(self, outputs, targets):
        pred = torch.round(torch.sigmoid(outputs))
        correct = (pred == targets).sum().float()
        acc = correct / targets.size(0)
        return acc.detach().cpu()
