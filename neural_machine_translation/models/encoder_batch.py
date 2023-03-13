import torch
from torchtext import data
from torchtext import datasets
import torch.nn.functional as F

class EncoderRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size 
 
        self.embedding = torch.nn.Embedding(input_size, hidden_size)
        self.gru = torch.nn.GRU(hidden_size, hidden_size, batch_first=True)
        
    def forward(self, input_index, hidden=None):
        output = self.embedding(input_index)
        if hidden is None:
            output, hidden = self.gru(output) 
        else:
            output, hidden = self.gru(output, hidden)
        return output, hidden
    