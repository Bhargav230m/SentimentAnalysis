import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# train_150k.txt is not included in this repo

df = pd.read_csv('train_150k.txt', sep='\t', names=('label', 'example'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

def apply_sentiment(index):
    sentiment = ["negative", "positive"]
    return sentiment[index]

df['example'] = df['example'].apply(preprocess_text)
df['label'] = df['label'].apply(apply_sentiment)

df.to_parquet("train_preprocessed.parquet", index=False)
