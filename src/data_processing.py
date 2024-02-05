import json
import torch

class DataLoader:
    def tokenize_sentence(oracion, word_to_index):
        return [word_to_index.get(word, 0) for word in oracion.split()]

    def load_data(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        word_to_index_es = {}
        index_to_word_en = {}

        all_input_text = [item["es"] for item in data]
        all_target_textos = [item["en"] for item in data]

        index_data = []
        
        # Obtener las longitudes máximas de las oraciones
        max_input_length = max(len(oracion.split()) for oracion in all_input_text)
        max_target_length = max(len(oracion.split()) for oracion in all_target_textos)

        for input_texto, target_texto in zip(all_input_text, all_target_textos):
            input_indices = DataLoader.tokenize_sentence(input_texto, word_to_index_es)
            target_indices = DataLoader.tokenize_sentence(target_texto, index_to_word_en)

            # Paddear las oraciones para que tengan la misma longitud
            input_indices = torch.tensor(input_indices + [0] * (max_input_length - len(input_indices)))
            target_indices = torch.tensor(target_indices + [0] * (max_target_length - len(target_indices)))

            index_data.append((input_indices, target_indices))

        return index_data, word_to_index_es, index_to_word_en