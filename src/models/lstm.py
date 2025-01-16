import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_dim, output_dim ,hidden_dim = 64, layer_dim = 2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.h0 = None
        self.c0 = None

    def predict_one_stateful(self, x):
        o, h0, c0 = self.forward(x, self.h0, self.c0, return_states=True)
        self.h0 = h0
        self.c0 = c0
        return o

    def forward(self, x, h0=None, c0=None, return_states = False):
        if h0 is None:
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, requires_grad=True)
        if c0 is None:
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, requires_grad=True)
        
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :]) 
        if return_states:
            return out, h0, c0
        else:
            return out