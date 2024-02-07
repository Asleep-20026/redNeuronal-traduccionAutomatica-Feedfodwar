
import torch.nn as nn

class WordProcessorModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, output_size):
        super(WordProcessorModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        lstm_last = lstm_out[:, -1, :]
        output = self.fc(lstm_last)
        return output
