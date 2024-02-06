from inference import *
from model import *
from train import *
from data_processing import *

# Parámetros de entrenamiento
vocab_size = 10000  
embedding_size = 200
hidden_size = 200
output_size = 10000  

# Crear modelo
model = WordProcessorModel(vocab_size, embedding_size, hidden_size, output_size)

DataLoader.index_data, _, _ = DataLoader.load_data(Trainer.dataTrain)
Trainer.train_model(model, DataLoader.index_data, output_size)

# Ejemplo de uso
new_sentence = "Camarón que se duerme, se lo lleva la corriente?"
print(f"Oración original: {new_sentence}")
print(f"Traducción (índices): {Translator.translation}")
print(f"Salida del modelo: {Translator.output}")   