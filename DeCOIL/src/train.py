import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

# Define dimensions
N_OBJECTIVES = 9  # Number of objectives
INPUT_DIM = 160   # 160D input space
HIDDEN_DIM = 256  # Hidden layer size
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
ENSEMBLE_SIZE = 5  # Number of models in the ensemble
SAVE_DIR = "saved_models"  # Directory to save models

# ----- Multitask Neural Network -----
class MultiTaskNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_objectives):
        super(MultiTaskNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_objectives)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # Output shape: (batch_size, N_OBJECTIVES)
        return x

# ----- Custom Dataset -----
class MultiObjectiveDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# ----- Generate Dummy Data (Replace with real data) -----
def generate_data(num_samples=1000, input_dim=INPUT_DIM, num_objectives=N_OBJECTIVES):
    X = np.random.rand(num_samples, input_dim)  # Random inputs
    Y = np.random.rand(num_samples, num_objectives)  # Random objective values
    return X, Y

# ----- Train One Model -----
def train_model(model, train_loader, val_loader, model_idx):
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_Y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_Y in val_loader:
                preds = model(batch_X)
                val_loss += criterion(preds, batch_Y).item()

        print(f"Model {model_idx+1}/{ENSEMBLE_SIZE} | Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")

    return model

# ----- Train an Ensemble of Models -----
def train_ensemble():
    # Generate dataset
    X_train, Y_train = generate_data(num_samples=5000)
    X_val, Y_val = generate_data(num_samples=1000)

    # Create DataLoaders
    train_dataset = MultiObjectiveDataset(X_train, Y_train)
    val_dataset = MultiObjectiveDataset(X_val, Y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Create directory to save models
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Train each model in the ensemble
    for i in range(ENSEMBLE_SIZE):
        print(f"\n--- Training Model {i+1}/{ENSEMBLE_SIZE} ---\n")
        model = MultiTaskNN(INPUT_DIM, HIDDEN_DIM, N_OBJECTIVES)
        trained_model = train_model(model, train_loader, val_loader, i)

        # Save the trained model
        model_path = os.path.join(SAVE_DIR, f"model_{i}.pt")
        torch.save(trained_model.state_dict(), model_path)
        print(f"Model {i+1} saved at {model_path}")

# Run training
if __name__ == "__main__":
    train_ensemble()