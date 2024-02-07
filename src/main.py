from inference import *
from model import *
from train import *
from data_processing import *

class Main:
    def __init__(self):
        self.index_data, _, _ = DataLoader.load_data(Trainer.dataTrain)
        self.model = WordProcessorModel(Trainer.vocab_size, Trainer.embedding_size, Trainer.hidden_size, Trainer.output_size)
        self.train_model()
        self.translate_example("Camarón que se duerme, se lo lleva la corriente?")

    def train_model(self):
        # Entrenar el modelo
        Trainer.train_model(self.model, self.index_data, Trainer.output_size)


# Crear una instancia de Main y comenzar la aplicación
start = Main()
Translator.translate_sentence()