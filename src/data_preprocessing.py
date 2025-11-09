# src/data_preprocessing.py
import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data once
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    return " ".join([word for word in text.split() if word not in stop_words])

def removing_numbers(text):
    return ''.join([char for char in text if not char.isdigit()])

def lower_case(text):
    return " ".join([word.lower() for word in text.split()])

def removing_punctuations(text):
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛', "")
    text = re.sub('\s+', ' ', text)
    return " ".join(text.split()).strip()

def removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub('', text)

def normalize_text(text_series):
    """Apply all preprocessing steps to a pandas Series."""
    return (
        text_series
        .apply(lower_case)
        .apply(remove_stop_words)
        .apply(removing_numbers)
        .apply(removing_punctuations)
        .apply(removing_urls)
        .apply(lemmatization)
    )

def main():
    # Load data
    train_data = pd.read_csv("data/raw/train.csv")
    test_data = pd.read_csv("data/raw/test.csv")
    
    # Preprocess
    train_data["content"] = normalize_text(train_data["content"])
    test_data["content"] = normalize_text(test_data["content"])
    
    # Save
    os.makedirs("data/processed", exist_ok=True)
    train_data.to_csv("data/processed/train_processed.csv", index=False)
    test_data.to_csv("data/processed/test_processed.csv", index=False)

if __name__ == "__main__":
    main()