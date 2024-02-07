from inference import *
from model import *
from train import *
from data_processing import *

# Crear modelo
DataLoader.index_data, word_to_index_es, _ = DataLoader.load_data(Trainer.dataTrain)
model = WordProcessorModel(Trainer.vocab_size, Trainer.embedding_size, Trainer.hidden_size, Trainer.output_size)
model.load_state_dict(torch.load(Trainer.model_path))
model.eval()
Trainer.train_model(model, DataLoader.index_data, Trainer.output_size)
# Ejemplo de uso
new_sentence = "Camarón que se duerme, se lo lleva la corriente?"
print(f"Oración original: {new_sentence}")
print(f"Traducción (índices): {Translator.translation}")
print(f"Salida del modelo: {Translator.output}")   