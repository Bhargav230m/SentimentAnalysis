from model import Classifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
import pickle

def save_hyperparams(dict):
    with open("hyperparams.hparams", "wb") as f:
        pickle.dump(dict, f)

# Load dataset
class ParquetDataset(Dataset):
    def __init__(self, file_path, vocab_=None):
        self.data = pd.read_parquet(file_path)
        self.vectorizer = CountVectorizer()
        self.vectorizer.fit(self.data["example"])
        self.input_size = len(self.vectorizer.vocabulary_)
        self.labels = self.data["label"].unique()
        self.num_labels = len(self.labels)
        if vocab_:
            self.load_vocab(vocab_)
        else:
            self.vectorizer.fit(self.data["example"])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        label = 1 if sample["label"] == "positive" else 0
        example = sample["example"]
        example_vec = self.vectorizer.transform([example]).toarray().squeeze()
        return example_vec, label
    
    def save_vocab(self, vocab_file):
        with open(vocab_file, 'wb') as f:
            pickle.dump(self.vectorizer.vocabulary_, f)
    
    def load_vocab(self, vocab_file):
        with open(vocab_file, 'rb') as f:
            vocab = pickle.load(f)
            self.vectorizer.vocabulary_ = vocab
            self.vectorizer._validate_vocabulary()

batch_size = 64

dataset = ParquetDataset("data/train_preprocessed.parquet")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

learning_rate = 0.0001
input_size = dataset.input_size
num_epochs = 5
hidden_size = 10
output_size = dataset.num_labels
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Saving hyperparameters")
save_hyperparams(
    {
        "lr": learning_rate,
        "input_size": input_size,
        "num_epochs": num_epochs,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "labels": dataset.labels,
        "batch_size": batch_size,
        "device": device
    }
)
print("Saving vocabulary")
dataset.save_vocab("vocabulary.vocab")

# Define loss function and optimizer
model = Classifier(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    with tqdm(total=len(dataloader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs.float())

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.update(1)
            pbar.set_postfix(loss=f'{loss.item():.4f}')

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}')

print("Training complete")
torch.save(model.state_dict(), "model.pt")
print("Saved Model")