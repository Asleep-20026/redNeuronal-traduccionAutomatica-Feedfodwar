import torch
from model import WordProcessorModel  
from data_processing import DataLoader
from train import *

class Translator:

    def translate_sentence(oracion, model, word_to_index_es):
        with torch.no_grad():
            input_indices = DataLoader.tokenize_sentence(oracion, word_to_index_es)
            input_tensor = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0)
            output = model(input_tensor)
            _, predicted_index = torch.max(output, 1)
            predicted_index = predicted_index.item()  
            print(f"Índice predicho: {predicted_index}")
            translation = word_to_index_es.get(predicted_index, "Palabra Desconocida")
            print(f"Traducción: {translation}")

        return translation, output