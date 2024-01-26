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

        # Repetir hidden state a lo largo de las dimensiones de la secuencia
        hidden = hidden.repeat(seq_len, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)

        # Calcular las puntuaciones de atención
        energy = F.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        energy = energy.transpose(1, 2)  # Cambiar las dimensiones para la multiplicación matricial
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)
        attention_scores = torch.bmm(v, energy).squeeze(1)

        # Aplicar softmax para obtener los pesos de atención
        attention_weights = F.softmax(attention_scores, dim=1)

        # Aplicar ponderaciones de atención a los estados codificados
        context_vector = torch.bmm(attention_weights.unsqueeze(0),
                                   encoder_outputs.transpose(0, 1)).squeeze(0)

        return context_vector, attention_weights

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

    def forward(self, x, encoder_outputs):
        # Aplicar embedding con dropout
        embedded = self.embedding(x)
        embedded = self.embedding_dropout(embedded)

        # Aplicar capa LSTM con dropout
        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.lstm_dropout(lstm_out)

        # Aplicar atención
        context_vector, attention_weights = self.attention(lstm_out[:, -1, :], encoder_outputs)

        # Concatenar el contexto a la salida de la LSTM
        lstm_with_attention = torch.cat((lstm_out[:, -1, :], context_vector), dim=1)

        # Aplicar capa lineal de salida con dropout
        output = self.fc(lstm_with_attention)
        output = self.fc_dropout(output)

        return output
