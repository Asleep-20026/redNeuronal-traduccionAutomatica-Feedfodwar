import torch
import torch.optim as optim
import torch.nn as nn
from model import WordProcessorModel
from data_processing import cargar_datos

class Trainer:
    def __init__(self, model, criterion, optimizer, data_path, num_epochs=10):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.data_indices, _, _ = cargar_datos(data_path)
        self.num_epochs = num_epochs

    def train(self):
        for epoch in range(self.num_epochs):
            total_loss = 0

            for input_indices, target_indices in self.data_indices:
                input_tensor = torch.tensor([int(idx) if idx.isdigit() else 0 for idx in input_indices], dtype=torch.long)
                target_tensor = torch.tensor([int(idx) if idx.isdigit() else 0 for idx in target_indices], dtype=torch.long)

                self.optimizer.zero_grad()

                _, (encoder_outputs, _) = self.model.lstm(self.model.embedding(input_tensor.unsqueeze(1)))
                output = self.model(input_tensor.unsqueeze(1), encoder_outputs)

                input_len = input_tensor.size(0)
                target_len = target_tensor.size(0)

                if input_len < target_len:
                    target_tensor = target_tensor[:input_len]
                elif input_len > target_len:
                    padding = torch.zeros(input_len - target_len, dtype=torch.long)
                    target_tensor = torch.cat((target_tensor, padding))

                loss = self.criterion(output.view(-1, self.model.output_size), target_tensor.view(-1))

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f'Epoch {epoch + 1}/{self.num_epochs}, Loss: {total_loss / len(self.data_indices)}')

        torch.save(self.model.state_dict(), 'trained_model.pth')

# Parámetros de entrenamiento
vocab_size = 10000
embedding_size = 200
hidden_size = 200
output_size = 10000

# Crear modelo
model = WordProcessorModel(vocab_size, embedding_size, hidden_size, output_size)

# Definir función de pérdida y optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Crear instancia de Trainer
trainer = Trainer(model, criterion, optimizer, dataTrain, num_epochs=10)

# Entrenar el modelo
trainer.train()
