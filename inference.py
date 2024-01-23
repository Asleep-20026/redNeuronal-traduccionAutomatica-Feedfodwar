# En un archivo llamado inference.py
import torch
from model import WordProcessorModel
from data_processing import tokenizar_oracion
from model import dataTrain
from data_processing import cargar_datos

datos, word_to_index_es, index_to_word_en = cargar_datos(dataTrain)

def cargar_modelo(ruta_modelo, modelo_clase, vocab_size, embedding_size, hidden_size, output_size):
    
    model = modelo_clase(vocab_size, embedding_size, hidden_size, output_size)
    
    model.load_state_dict(torch.load(ruta_modelo))
    
    model.eval()
    
    return model

def traducir_oracion(model, input_sentence, tokenizar_fn):
    # Tokenizar y convertir a índices la oración de entrada
    input_tokens = tokenizar_fn(input_sentence)
    input_indices = [word_to_index_es.get(palabra, 0) for palabra in input_tokens]
    
    input_tensor = torch.tensor(input_indices, dtype=torch.long)

    with torch.no_grad():
        output = model(input_tensor.unsqueeze(1))  # Agregar dimensión de secuencia (batch_size=1)

    # Convertir los índices de salida a palabras
    output_tokens = [index_to_word_en[int(idx)] for idx in output.argmax(dim=-1).tolist()]

    output_sentence = " ".join(output_tokens)

    return output_sentence

# Cargar el modelo  
ruta_modelo_entrenado = 'modelo_entrenado.pth'
modelo = cargar_modelo(
    ruta_modelo_entrenado,
    WordProcessorModel,
    vocab_size=10000,
    embedding_size=200,
    hidden_size=200,
    output_size=10000
)

# Elegir una oración para traducir
input_sentence = input("Introduce una oración en español para traducir: ")

# Realizar la traducción
traduccion = traducir_oracion(modelo, input_sentence, tokenizar_oracion)
    
print("Oración original en español:", input_sentence)
print("Traducción al inglés:", traduccion)
