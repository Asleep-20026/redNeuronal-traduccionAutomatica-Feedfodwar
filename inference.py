import torch
from model import WordProcessorModel
from data_processing import tokenizar_oracion, cargar_datos

class Translator:
    def __init__(self, model_path, vocab_size, embedding_size, hidden_size, output_size):
        self.model = self._cargar_modelo(model_path, vocab_size, embedding_size, hidden_size, output_size)

    def _cargar_modelo(self, ruta_modelo, vocab_size, embedding_size, hidden_size, output_size):
        model = WordProcessorModel(vocab_size, embedding_size, hidden_size, output_size)
        model.load_state_dict(torch.load(ruta_modelo))
        model.eval()
        return model

    def traducir_oracion(self, input_sentence):
        input_tokens = tokenizar_oracion(input_sentence)
        input_indices = [word_to_index_es.get(palabra, 0) for palabra in input_tokens]
        input_tensor = torch.tensor(input_indices, dtype=torch.long)

        with torch.no_grad():
            output = self.model(input_tensor.unsqueeze(1))

        output_tokens = [index_to_word_en.get(int(idx), 'UNK') for idx in output.argmax(dim=-1).tolist()]
        output_sentence = " ".join(output_tokens)

        return output_sentence

if __name__ == "__main__":
    dataTrain = "./dataTrain.json"
    datos, word_to_index_es, index_to_word_en = cargar_datos(dataTrain)

    translator = Translator(
        model_path='modelo_entrenado.pth',
        vocab_size=10000,
        embedding_size=200,
        hidden_size=200,
        output_size=10000
    )

    input_sentence = input("Introduce una oración en español para traducir: ")
    traduccion = translator.traducir_oracion(input_sentence)

    print("Oración original en español:", input_sentence)
    print("Traducción al inglés:", traduccion)
