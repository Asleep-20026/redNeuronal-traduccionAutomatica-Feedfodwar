# train.py
import torch
import torch.optim as optim
import torch.nn as nn
from model import WordProcessorModel
from data_processing import cargar_datos, tokenizar_oracion

def train_model(model, datos_indices, num_epochs=10, learning_rate=0.001, output_size=10000):
    # Definir función de pérdida y optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0

        # Iterar sobre los pares de oraciones (entrada, salida)
        for input_indices, target_indices in datos_indices:
            # Crear tensores de entrada y salida
            input_tensor = torch.tensor(input_indices, dtype=torch.long)
            target_tensor = torch.tensor(target_indices, dtype=torch.long)

            # Resetear gradientes
            optimizer.zero_grad()

            # Realizar forward pass
            output = model(input_tensor.unsqueeze(1))  # Agregar dimensión de secuencia (batch_size=1)

            # Ajustar las dimensiones de la salida
            output = output.view(-1, output_size)[:target_tensor.size(0), :]

            # Calcular pérdida
            loss = criterion(output, target_tensor.view(-1)[:output.size(0)])

            # Realizar backward pass y optimización
            loss.backward()
            optimizer.step()

            # Acumular la pérdida total
            total_loss += loss.item()

        # Imprimir la pérdida promedio en cada época
        print(f'Época {epoch + 1}/{num_epochs}, Pérdida: {total_loss / len(datos_indices)}')
        torch.save(model.state_dict(), 'modelo_entrenado.pth')


# Parámetros de entrenamiento
vocab_size = 10000  
embedding_size = 200
hidden_size = 200
output_size = 10000  

# Crear modelo
model = WordProcessorModel(vocab_size, embedding_size, hidden_size, output_size)

# Ruta del archivo de datos
dataTrain = "/mnt/c/workspace/redNeuronal-traduccionAutomatica-Feedfodwar/data/dataTrain.json"

# Cargar datos de entrenamiento
datos_indices, _, _ = cargar_datos(dataTrain)

# Entrenar el modelo
train_model(model, datos_indices, output_size)
