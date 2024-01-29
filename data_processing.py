import json

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.word_to_index_es = {}
        self.index_to_word_en = {}
        self.data = None

    def tokenize_sentence(self, oracion):
        return oracion.split()

    def load_data(self):
        with open(self.file_path, 'r', encoding='utf-8') as file:
            self.data = json.load(file)

        all_input_text = [item["es"] for item in self.data]
        all_target_textos = [item["en"] for item in self.data]

        for input_texto in all_input_text:
            tokens = self.tokenize_sentence(input_texto)
            for token in tokens:
                if token not in self.word_to_index_es:
                    self.word_to_index_es[token] = len(self.word_to_index_es) + 1

        for target_texto in all_target_textos:
            tokens = self.tokenize_sentence(target_texto)
            for token in tokens:
                if token not in self.index_to_word_en:
                    self.index_to_word_en[len(self.index_to_word_en)] = token

    def build_vocabularies(self):
        if self.data is None:
            self.load_data()

    def get_data_and_vocabularies(self):
        if self.data is None:
            self.load_data()
        return self.data, self.word_to_index_es, self.index_to_word_en

# Uso de la clase DataLoader
data_loader = DataLoader(file_path="./dataTrain.json")
data_loader.build_vocabularies()
data, word_to_index_es, index_to_word_en = data_loader.get_data_and_vocabularies()