import json

dataTrain = "./dataTrain.json"

def tokenizar_oracion(oracion):

    return oracion.split()

def cargar_datos(dataTrain):
    with open(dataTrain, 'r', encoding='utf-8') as file:
        datos = json.load(file)

    word_to_index_es = {}
    index_to_word_en = {}
    all_input_textos = [item["es"] for item in datos]  # Obtener todas las oraciones en español
    all_target_textos = [item["en"] for item in datos]  # Obtener todas las oraciones en inglés

    # Construir el diccionario word_to_index_es
    for input_texto in all_input_textos:
        tokens = tokenizar_oracion(input_texto)
        for token in tokens:
            if token not in word_to_index_es:
                word_to_index_es[token] = len(word_to_index_es) + 1  # Asignar un índice único

    # Construir el diccionario index_to_word_en
    for target_texto in all_target_textos:
        tokens = tokenizar_oracion(target_texto)
        for token in tokens:
            if token not in index_to_word_en:
                index_to_word_en[len(index_to_word_en)] = token  # Asignar un índice único

    # Devolver los datos junto con los diccionarios word_to_index_es e index_to_word_en
    return datos, word_to_index_es, index_to_word_en