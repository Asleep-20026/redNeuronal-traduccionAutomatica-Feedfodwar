# En un archivo llamado train.py
import torch
import torch.optim as optim
import torch.nn as nn
from model import WordProcessorModel
from data_processing import cargar_datos
from model import dataTrain

# Parámetros de entrenamiento
vocab_size = 10000  # Ajusta según tu vocabulario
embedding_size = 200
hidden_size = 200
output_size = 10000  # Ajusta según tus necesidades de salida

# Crear modelo
model = WordProcessorModel(vocab_size, embedding_size, hidden_size, output_size)

# Definir función de pérdida y optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Cargar datos de entrenamiento (ajusta la ruta del archivo según tu caso)
datos_indices = cargar_datos(dataTrain)

# Número de épocas (iteraciones completas sobre los datos de entrenamiento)
num_epochs = 10

# Bucle de entrenamiento
for epoch in range(num_epochs):
    total_loss = 0

    # Iterar sobre los pares de oraciones (entrada, salida)
    for input_indices, target_indices in datos_indices:
        # Crear tensores de entrada y salida
        input_tensor = torch.tensor([int(idx) if idx.isdigit() else 0 for idx in input_indices], dtype=torch.long)
        target_tensor = torch.tensor([int(idx) if idx.isdigit() else 0 for idx in target_indices], dtype=torch.long)

        # Resetear gradientes
        optimizer.zero_grad()

        # Realizar forward pass
        output = model(input_tensor.unsqueeze(1))  # Agregar dimensión de secuencia (batch_size=1)

        # Determinar la longitud de las secuencias de entrada y salida
        input_len = input_tensor.size(0)
        target_len = target_tensor.size(0)

        # Truncar o rellenar la salida y los objetivos
        if input_len < target_len:
            target_tensor = target_tensor[:input_len]
        elif input_len > target_len:
            padding = torch.zeros(input_len - target_len, dtype=torch.long)
            target_tensor = torch.cat((target_tensor, padding))

        # Calcular pérdida
        loss = criterion(output.view(-1, output_size), target_tensor.view(-1))

        # Realizar backward pass y optimización
        loss.backward()
        optimizer.step()

        # Acumular la pérdida total
        total_loss += loss.item()

    # Imprimir la pérdida promedio en cada época
    print(f'Época {epoch + 1}/{num_epochs}, Pérdida: {total_loss / len(datos_indices)}')
    
# Guardar los pesos entrenados del modelo
torch.save(model.state_dict(), 'modelo_entrenado.pth')
