from model import Classifier
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import torch
import torch.nn.functional as F

def load_vocab(vocab_file):
    with open(vocab_file, 'rb') as f:
        vocab = pickle.load(f)
    vectorizer = CountVectorizer(vocabulary=vocab)
    return vectorizer

def load_model(model_file, input_size, hidden_size, output_size):
    model = Classifier(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(model_file, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    model.eval()
    return model

def load_hyperparams(file):
    with open(file, "rb") as f:
        h = pickle.load(f)
    return h

def classify_text(model, vectorizer, text, labels):
    example_vec = vectorizer.transform([text]).toarray()
    example_vec = torch.tensor(example_vec, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(example_vec)
        outputs = F.softmax(outputs, dim=1)
        probabilities = outputs.squeeze().tolist()
    return {label: prob for label, prob in zip(labels, probabilities)}

vectorizer = load_vocab("model/vocabulary.vocab")
hyperparams = load_hyperparams("model/hyperparams.hparams")
input_size = len(vectorizer.vocabulary)
hidden_size = hyperparams['hidden_size']
output_size = hyperparams['output_size']
labels = hyperparams['labels']
model = load_model("model/model.pt", input_size, hidden_size, output_size)

while True:
    user_input = input("Enter text to classify: ")
    if(user_input) == "stop_": break

    classification_result = classify_text(model, vectorizer, user_input, labels)
    for label, prob in classification_result.items():
        print(f'{label}: {prob:.4f}')
