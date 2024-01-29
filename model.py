import torch
import torch.nn as nn
import torch.nn.functional as F

dataTrain = "./dataTrain.json"

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        self.v.data.normal_(mean=0, std=1. / self.v.size(0))

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(1)

        # Expandir hidden para que tenga las mismas dimensiones que encoder_outputs
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)

        # Transponer encoder_outputs para que las dimensiones coincidan
        encoder_outputs = encoder_outputs.transpose(1, 2)

        # Calcular las puntuaciones de atención
        energy = F.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # Transponer energy de nuevo
        energy = energy.transpose(1, 2)

        # Calcular pesos de atención
        attention_scores = F.softmax(torch.matmul(self.v.unsqueeze(0), energy), dim=2)

        # Aplicar ponderaciones de atención a los estados codificados
        context_vector = torch.bmm(attention_scores, encoder_outputs.transpose(1, 2)).squeeze(1)

        return context_vector, attention_scores

class WordProcessorModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, output_size, dropout_prob=0.5):
        super(WordProcessorModel, self).__init__()

        # Capa de embedding con dropout
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embedding_dropout = nn.Dropout(dropout_prob)

        # Capa LSTM con dropout
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.lstm_dropout = nn.Dropout(dropout_prob)

        # Capa de atención
        self.attention = Attention(hidden_size)

        # Capa lineal de salida con dropout
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Doble del tamaño debido a la concatenación
        self.fc_dropout = nn.Dropout(dropout_prob)

    def forward(self, x, hidden=None):
        # Aplicar embedding con dropout
        embedded = self.embedding(x)
        embedded = self.embedding_dropout(embedded)

        # Aplicar capa LSTM con dropout
        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.lstm_dropout(lstm_out)

        # Tomar solo el estado final de la LSTM como encoder_outputs si no se proporciona
        encoder_outputs = hidden if hidden is not None else lstm_out  # Cambiado aquí

        # Aplicar atención
        context_vector, attention_weights = self.attention(lstm_out[:, -1, :], encoder_outputs)

        # Concatenar el contexto a la salida de la LSTM
        lstm_with_attention = torch.cat((lstm_out[:, -1, :], context_vector), dim=1)

        # Aplicar capa lineal de salida con dropout
        output = self.fc(lstm_with_attention)
        output = self.fc_dropout(output)

        return output, (lstm_out, _)  # Devolver también el estado interno de la LSTM si es necesario
