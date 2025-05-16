import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import mlflow
import numpy as np

DATA_DIR = "data/raw"
OUTPUT_DIR = "data/processed"
TARGET_CLASSES = ["BENIGN", "DoS Hulk", "DDoS", "PortScan"]

def load_dataset():
    all_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    df_list = []

    for file in all_files:
        print(f"Reading {file}")
        df = pd.read_csv(os.path.join(DATA_DIR, file), low_memory=False)
        df_list.append(df)

    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df

def main():
    mlflow.set_experiment("CyberThreatPreprocessing")
    with mlflow.start_run():
        mlflow.autolog()

        print("Loading data...")
        df = load_dataset()
        print("Initial shape:", df.shape)

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        print("After removing NaNs/Infs:", df.shape)

        if " Label" in df.columns:
            df.rename(columns={" Label": "Label"}, inplace=True)
        elif "Label" not in df.columns:
            raise ValueError("'Label' column not found in data!")

        df = df[df["Label"].isin(TARGET_CLASSES)]
        label_map = {"BENIGN": 0, "DoS Hulk": 1, "DDoS": 2, "PortScan": 3}
        df["Label"] = df["Label"].map(label_map)

        # Drop non-numeric columns just in case
        df = df.select_dtypes(include=["number"]).copy()

        X = df.drop("Label", axis=1)
        y = df["Label"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        df_scaled["Label"] = y.values

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, "cleaned_data.csv")
        df_scaled.to_csv(output_path, index=False)
        print(f"Saved cleaned & normalized data to {output_path}")

if __name__ == "__main__":
    main()
