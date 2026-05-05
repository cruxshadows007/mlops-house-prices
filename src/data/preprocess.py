import pandas as pd
import os

RAW_PATH = "data/raw/train.csv"
PROCESSED_PATH = "data/processed/train.csv"

def load_data():
    return pd.read_csv(RAW_PATH)

def clean_data(df):
    # eliminar columnas con demasiados nulos
    df = df.dropna(axis=1, thresh=int(0.8 * len(df)))

    # separar target si existe
    return df

def save_data(df):
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

def main():
    df = load_data()
    df = clean_data(df)
    save_data(df)
    print("Preprocessing completed")

if __name__ == "__main__":
    main()