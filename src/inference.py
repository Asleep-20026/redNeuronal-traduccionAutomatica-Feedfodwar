import torch
from model import WordProcessorModel  # Asegúrate de importar el modelo correctamente
from data_processing import tokenizar_oracion, cargar_datos  # Ajusta según tus funciones de procesamiento de datos

# Definir variables necesarias
vocab_size = 10000  # Ajusta según tu vocabulario
embedding_size = 200  # Ajusta según tu modelo de entrenamiento
hidden_size = 200  # Ajusta según tu modelo de entrenamiento
output_size = 10000  # Ajusta según tus necesidades de salida

# Cargar el modelo entrenado
model = WordProcessorModel(vocab_size, embedding_size, hidden_size, output_size)
model.load_state_dict(torch.load('modelo_entrenado.pth'))
model.eval()

# Cargar datos y obtener word_to_index_es
dataTrain = "/mnt/c/workspace/redNeuronal-traduccionAutomatica-Feedfodwar/data/dataTrain.json" # Ajusta la ruta según tu estructura de directorios
datos_indices, word_to_index_es, _ = cargar_datos(dataTrain)

def traducir_oracion(oracion, model, word_to_index_es):
    with torch.no_grad():
        input_indices = tokenizar_oracion(oracion, word_to_index_es)
        input_tensor = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0)
        output = model(input_tensor)

        # Obtener el índice del token con mayor probabilidad
        _, predicted_index = torch.max(output, 1)
        predicted_index = predicted_index.item()  # Obtener el valor entero del índice

        print(f"Índice predicho: {predicted_index}")

        # Convertir el índice de nuevo a palabra utilizando word_to_index_es
        traduccion = word_to_index_es.get(predicted_index, "Palabra Desconocida")

        print(f"Traducción: {traduccion}")

    return traduccion, output

# Ejemplo de uso
nueva_oracion = "Camarón que se duerme, se lo lleva la corriente?"
traduccion, output = traducir_oracion(nueva_oracion, model, word_to_index_es)
print(f"Oración original: {nueva_oracion}")
print(f"Traducción (índices): {traduccion}")
print(f"Salida del modelo: {output}")
