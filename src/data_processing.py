import json

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.word_to_index_es = {}
        self.index_to_word_en = {}
        self.data = None

    @staticmethod
    def tokenize_sentence(oracion):
        return oracion.split()

    @staticmethod
    def load_data(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        word_to_index_es = {}
        index_to_word_en = {}

        all_input_text = [item["es"] for item in data]
        all_target_textos = [item["en"] for item in data]

        for input_texto in all_input_text:
            tokens = DataLoader.tokenize_sentence(input_texto)
            for token in tokens:
                if token not in word_to_index_es:
                    word_to_index_es[token] = len(word_to_index_es) + 1

        for target_texto in all_target_textos:
            tokens = DataLoader.tokenize_sentence(target_texto)
            for token in tokens:
                if token not in index_to_word_en:
                    index_to_word_en[len(index_to_word_en)] = token

        return data, word_to_index_es, index_to_word_en

    def build_vocabularies(self):
        if self.data is None:
            self.data, self.word_to_index_es, self.index_to_word_en = self.load_data(self.file_path)

    def get_data_and_vocabularies(self):
        self.build_vocabularies()
        return self.data, self.word_to_index_es, self.index_to_word_en

# Uso de la clase DataLoader
data_loader = DataLoader(file_path="../data/dataTrain.json")
data_loader.build_vocabularies()
data, word_to_index_es, index_to_word_en = data_loader.get_data_and_vocabularies()