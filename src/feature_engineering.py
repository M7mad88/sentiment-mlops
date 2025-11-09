# src/feature_engineering.py
import os
import yaml
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def main():
    # Load params
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    
    # Load data
    train_data = pd.read_csv("data/processed/train_processed.csv")
    test_data = pd.read_csv("data/processed/test_processed.csv")
    
    # Handle missing values
    train_data['content'] = train_data['content'].fillna('')
    test_data['content'] = test_data['content'].fillna('')
    
    # Vectorize
    vectorizer = CountVectorizer(max_features=params["feature_engineering"]["max_features"])
    X_train_bow = vectorizer.fit_transform(train_data['content'])
    X_test_bow = vectorizer.transform(test_data['content'])
    
    # Create feature DataFrames
    train_df = pd.DataFrame(X_train_bow.toarray())
    train_df['label'] = train_data['sentiment']
    
    test_df = pd.DataFrame(X_test_bow.toarray())
    test_df['label'] = test_data['sentiment']
    
    # Save
    os.makedirs("data/features", exist_ok=True)
    train_df.to_csv("data/features/train_bow.csv", index=False)
    test_df.to_csv("data/features/test_bow.csv", index=False)

if __name__ == "__main__":
    main()