import torch
import torch.optim as optim
import torch.nn as nn
from model import WordProcessorModel
from data_processing import DataLoader

# Parámetros de entrenamiento
embedding_size = 200
hidden_size = 200
output_size = 10000  # Ajusta según tus necesidades de salida

# Crear instancia de DataLoader
data_loader = DataLoader(file_path="../data/dataTrain.json")

# Cargar datos de entrenamiento
data, word_to_index_es, index_to_word_en = data_loader.get_data_and_vocabularies()

# Obtener el tamaño del vocabulario del DataLoader
vocab_size = len(word_to_index_es)

# Crear modelo
model = WordProcessorModel(vocab_size, embedding_size, hidden_size, output_size)
# Definir función de pérdida y optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Número de épocas (iteraciones completas sobre los datos de entrenamiento)
num_epochs = 10

# Bucle de entrenamiento
for epoch in range(num_epochs):
    total_loss = 0

    # Iterar sobre los pares de oraciones (entrada, salida)
    for input_indices, target_indices in data:
        # Crear tensores de entrada y salida
        input_tensor = torch.tensor([word_to_index_es.get(idx, 0) for idx in input_indices], dtype=torch.long)
        target_tensor = torch.tensor([word_to_index_es.get(idx, 0) for idx in target_indices], dtype=torch.long)

        # Resetear gradientes
        optimizer.zero_grad()

        # Proporcionar la salida de la capa LSTM como encoder_outputs
        _, (encoder_outputs, _) = model.lstm(model.embedding(input_tensor.unsqueeze(1)))

        # Realizar forward pass
        output, _, attention_weights = model(input_tensor.unsqueeze(1), encoder_outputs)

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

    # Imprimir los pesos de atención después de cada época
    print("Pesos de atención después de la época", epoch + 1, ":", attention_weights)

    # Imprimir la pérdida promedio en cada época
    print(f'Época {epoch + 1}/{num_epochs}, Pérdida: {total_loss / len(data)}')

# Guardar los pesos entrenados del modelo
torch.save(model.state_dict(), 'modelo_entrenado.pth')
