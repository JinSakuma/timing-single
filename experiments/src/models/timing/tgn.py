import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TGN(nn.Module):

    def __init__(self, device, input_dim, hidden_dim): #, silence_encoding_type="concat"):
        super().__init__()
        
        self.device = device
        # self.silence_encoding_type = silence_encoding_type
        # if silence_encoding_type=="concat":
        #    input_dim += 1

        self.lstm = torch.nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                batch_first=True,
            )

        self.fc = nn.Linear(hidden_dim, 1)
        self.criterion = nn.MSELoss().to(self.device)

    def get_silence(self, uttr_label):
        silence = torch.zeros(len(uttr_label)).to(self.device)
        silence[0] = uttr_label[0]
        for i, u in enumerate(uttr_label):
            if u == 1:
                silence[i] = 0
            else:
                silence[i] = silence[i-1]+1
        return silence.unsqueeze(1)

    def get_alpha(self, inputs):
        inputs = inputs.unsqueeze(0)
        h, _ = self.lstm(inputs, None)
        alpha = torch.sigmoid(self.fc(h))

        return alpha  

    def forward(self, inputs, uttr_pred, timing_label, uttr_label, thres1=0.8, thres2=0.4, thres_u=0.5):
        #silences = self.get_silence(uttr_labels)
        #if self.silence_encoding_type=="concat":
        #    inputs = torch.cat([inputs, silences], dim=-1)
        #else:
        #    raise Exception('Not implemented')

        alpha = self.get_alpha(inputs)
        alpha = alpha.view(-1)  # (t)
        
        seq_len = alpha.size(0)
        a_pre, y_pre, u_pre = 0, 0, 0
        loss, loss_c, loss_e = 0, 0, 0
        a_pred = np.asarray([])
        u_pred = np.asarray([])
        y = np.asarray([])
        
        flg=False
        for i in range(1, seq_len):
            ui = 1-uttr_label[i]
            ui_pred = 1-uttr_pred[i]
            ai = alpha[i]
            target = timing_label[i-1]

            a_gated = ui_pred * a_pre + (1-ui_pred) * ai
            yi = a_gated * ui_pred + (1-a_gated) * y_pre

            if target > thres1 and not flg:
                flg = True
                if ui_pred < thres_u:
                    loss_c = -1
                else:
                    loss_c = self.criterion(yi, torch.tensor(1.0).to(self.device)*thres1)
                    label_frame = i

            a_pre = a_gated
            y_pre = yi
            a_pred = np.append(a_pred, a_gated.detach().cpu().numpy())
            u_pred = np.append(u_pred, ui_pred.detach().cpu().numpy())
            y = np.append(y, yi.detach().cpu().numpy())

        if loss_c==-1:
            loss_c = 0
            loss_e = 0
            loss = 0            
        elif loss_c != 0:
            loss = loss + loss_c
        elif yi >= thres1: # 正解のタイミングがない場合は閾値を超えていれば最適化
            loss_e = self.criterion(y_pre, torch.tensor(1).to(self.device)*thres2)
            loss = loss + loss_e
        
        return loss, (y, a_pred, u_pred)