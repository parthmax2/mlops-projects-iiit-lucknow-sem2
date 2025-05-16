import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import mlflow
import mlflow.pytorch
import numpy as np

mlflow.set_experiment("CyberThreatModelTraining")

# Model definition
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # 4 classes
        )

    def forward(self, x):
        return self.net(x)

def load_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns=["Label"]).values
    y = df["Label"].values

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    return random_split(dataset, [train_size, val_size])

def train_model(train_loader, val_loader, input_dim):
    model = SimpleNN(input_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/10], Loss: {total_loss:.4f}")

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy:.2f}%")
    return model, accuracy, train_loader

def main():
    with mlflow.start_run():
        mlflow.autolog()

        train_dataset, val_dataset = load_data("data/processed/cleaned_data.csv")
        train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1024)

        input_dim = train_dataset[0][0].shape[0]

        model, accuracy, train_loader = train_model(train_loader, val_loader, input_dim)

        # Prepare input_example (convert torch.Tensor to numpy array)
        input_example = train_dataset[0][0].unsqueeze(0).numpy()

        # Log validation accuracy and the trained model to MLflow
        mlflow.log_metric("val_accuracy", accuracy)
        mlflow.pytorch.log_model(model, "cyber_model", input_example=input_example)

        print(f"Model saved to MLflow with accuracy {accuracy:.2f}%")

if __name__ == "__main__":
    main()
