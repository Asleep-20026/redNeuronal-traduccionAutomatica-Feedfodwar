import torch
from model import WordProcessorModel  
from data_processing import DataLoader

class Translator:
    # Definir variables necesarias
    vocab_size = 10000  
    embedding_size = 200  #
    hidden_size = 200  
    output_size = 10000  

    model = WordProcessorModel(vocab_size, embedding_size, hidden_size, output_size)
    model.load_state_dict(torch.load('/mnt/c/workspace/redNeuronal-translationAutomatica-Feedfodwar/saved_models/model_trained.pth'))
    model.eval()

    dataTrain = "/mnt/c/workspace/redNeuronal-translationAutomatica-Feedfodwar/data/dataTrain.json" 
    index_data, word_to_index_es, _ = DataLoader.load_data(dataTrain)

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

# Ejemplo de uso
new_sentence = "Camarón que se duerme, se lo lleva la corriente?"
translation, output = Translator.translate_sentence(new_sentence, Translator.model, Translator.word_to_index_es)
print(f"Oración original: {new_sentence}")
print(f"Traducción (índices): {translation}")
print(f"Salida del modelo: {output}")   