
import json

import torch

def tokenizar_oracion(oracion, word_to_index):
    return [word_to_index.get(palabra, 0) for palabra in oracion.split()]

def cargar_datos(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    word_to_index_es = {}
    index_to_word_en = {}

    all_input_text = [item["es"] for item in data]
    all_target_textos = [item["en"] for item in data]

    datos_indices = []

    # Obtener las longitudes máximas de las oraciones
    max_input_length = max(len(oracion.split()) for oracion in all_input_text)
    max_target_length = max(len(oracion.split()) for oracion in all_target_textos)

    for input_texto, target_texto in zip(all_input_text, all_target_textos):
        input_indices = tokenizar_oracion(input_texto, word_to_index_es)
        target_indices = tokenizar_oracion(target_texto, index_to_word_en)

        # Paddear las oraciones para que tengan la misma longitud
        input_indices = torch.tensor(input_indices + [0] * (max_input_length - len(input_indices)))
        target_indices = torch.tensor(target_indices + [0] * (max_target_length - len(target_indices)))

        datos_indices.append((input_indices, target_indices))

    return datos_indices, word_to_index_es, index_to_word_en