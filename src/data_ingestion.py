# src/data_ingestion.py
import os
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

def load_data(data_url: str) -> pd.DataFrame:
    """Load CSV data from URL."""
    try:
        df = pd.read_csv(data_url)
        return df
    except Exception as e:
        print(f"Error loading data from {data_url}: {e}")
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter and encode sentiment labels."""
    try:
        df = df.drop(columns=['tweet_id'], errors='ignore')
        df = df[df['sentiment'].isin(['happiness', 'sadness'])]
        df['sentiment'] = df['sentiment'].map({'happiness': 1, 'sadness': 0})
        return df
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, output_dir: str):
    """Save train/test splits to disk."""
    os.makedirs(output_dir, exist_ok=True)
    train_data.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test_data.to_csv(os.path.join(output_dir, "test.csv"), index=False)

def main():
    # Load parameters
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    
    # Load and preprocess
    df = load_data(params["data_ingestion"]["url"])
    final_df = preprocess_data(df)
    
    # Split
    train_data, test_data = train_test_split(
        final_df,
        test_size=params["data_ingestion"]["test_size"],
        random_state=params["data_ingestion"]["random_state"],
        stratify=final_df["sentiment"]
    )
    
    # Save
    save_data(train_data, test_data, "data/raw")

if __name__ == "__main__":
    main()