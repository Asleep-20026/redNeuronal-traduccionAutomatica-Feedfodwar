import torch
import torch.optim as optim
import os
import torch.nn as nn
from model import WordProcessorModel
from data_processing import DataLoader

class Trainer:
    def train_model(model, index_data, num_epochs=10, learning_rate=0.001, output_size=10000):
        
        # Verificar si hay una GPU disponible
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Mover el modelo a la GPU si está disponible
        model = model.to(device)
        
        # Definir función de pérdida y optimizador
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        # Construir la ruta del archivo dataTrain.json
        
        for epoch in range(num_epochs):
            total_loss = 0
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, '..', 'saved_models', 'model_trained.pth')
            # Iterar sobre los pares de oraciones (entrada, salida)
            for input_indices, target_indices in index_data:
                input_tensor = torch.tensor(input_indices, dtype=torch.long)
                target_tensor = torch.tensor(target_indices, dtype=torch.long)
                optimizer.zero_grad()
                output = model(input_tensor.unsqueeze(1))  
                output = output.view(-1, output_size)[:target_tensor.size(0), :]
                loss = criterion(output, target_tensor.view(-1)[:output.size(0)])
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            print(f'Época {epoch + 1}/{num_epochs}, Pérdida: {total_loss / len(index_data)}')
            torch.save(model.state_dict(), model_path)

            # Verificar si la pérdida es igual a 0 y detener el entrenamiento
            if total_loss == 0.0:
                print("La pérdida alcanzó 0.0. Deteniendo el entrenamiento.")
                break