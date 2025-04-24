import torch
from torch import nn
from sklearn.model_selection import train_test_split
import json
import numpy as np

class Echo_model(nn.Module):
    def __init__(self):
        super().__init__()
        # Create 2 nn.Linear layers
        self.layer1 = nn.Linear(in_features=2, out_features=5)
        self.layer2 = nn.Linear(in_features=5, out_features=1)

    def forward(self, x):
        return self.layer2(self.layer1(x))

    def training_split(self, json_path):
        # Load data from JSON file
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract features and targets (ensure your JSON structure matches this)
        X_data = np.array(data["features"])
        y_data = np.array(data["targets"]).reshape(-1, 1)  # Ensure y has shape (N, 1)

        # Convert to PyTorch tensors
        X_torch_data = torch.from_numpy(X_data).float()
        y_torch_data = torch.from_numpy(y_data).float()

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_torch_data, y_torch_data, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def training_loop(self, x_train, y_train, x_test, y_test, epochs=1000, learning_rate=0.01):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {device} device")

        self.to(device)
        x_train, y_train = x_train.to(device), y_train.to(device)
        x_test, y_test = x_test.to(device), y_test.to(device)

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            self.train()

            # Forward pass
            y_pred = self(x_train)

            # Compute loss
            loss = loss_fn(y_pred, y_train)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Evaluate
            self.eval()
            with torch.inference_mode():
                test_pred = self(x_test)
                test_loss = loss_fn(test_pred, y_test)

            if epoch % 10 == 0:
                print(f"Epoch: {epoch} | Loss: {loss.item():.4f} | Test loss: {test_loss.item():.4f}")