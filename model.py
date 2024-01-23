import torch.nn as nn
dataTrain = "./dataTrain.json"
class WordProcessorModel(nn.Module):
        
    def __init__(self, vocab_size, embedding_size, hidden_size, output_size):
        super(WordProcessorModel, self).__init__()
        
        # Capa de embedding
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        
        # Capa LSTM
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        
        # Capa lineal de salida
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        
        # Aplicar embedding
        embedded = self.embedding(x)
        
        # Aplicar capa LSTM
        lstm_out, _ = self.lstm(embedded)
        
        # Tomar la salida de la última secuencia
        lstm_last = lstm_out[:, -1, :]
        
        # Aplicar capa lineal de salida
        output = self.fc(lstm_last)
        
        return output
