import torch
from torchtext import data
from torchtext import datasets
import torch.nn.functional as F

class DecoderRNN(torch.nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        
        self.embedding = torch.nn.Embedding(output_size, hidden_size)
        self.gru = torch.nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = torch.nn.Linear(hidden_size, output_size)
        
    def forward(self, input_index, hidden):
        output = self.embedding(input_index)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden
     