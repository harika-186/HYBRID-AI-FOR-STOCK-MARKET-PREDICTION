import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2Model
import os
from sklearn.model_selection import train_test_split
import re
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# --- Data Preparation ---
def prepare_data(features, label, sequence_length):
    """Prepares time-series data for training."""
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(features)
    scaler1 = MinMaxScaler()
    label_data = scaler1.fit_transform(label)
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i + sequence_length])
        y.append(label_data[i + sequence_length])  # Predict the first feature (e.g., closing price)
    return np.array(X), np.array(y), scaler, scaler1

class TimeSeriesDataset(Dataset):
    """Dataset for time-series data."""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- QINN Layer (Simplified Example) ---
class QINNLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(QINNLayer, self).__init__()
        self.weights = nn.Parameter(torch.rand(input_size, output_size))
        # Simplified "quantum-inspired" activation function
        self.activation = lambda x: torch.sin(x)

    def forward(self, x):
        return self.activation(torch.matmul(x, self.weights))

# --- QINN Model ---
class QINN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QINN, self).__init__()
        self.qinn1 = QINNLayer(input_size, hidden_size)
        self.qinn2 = QINNLayer(hidden_size, output_size)
        self.output_layer = nn.Linear(output_size, 1)

    def forward(self, x):
        x = self.qinn1(x)
        x = self.qinn2(x)
        x = self.output_layer(x)
        return x.squeeze()

dataset = pd.read_csv("Dataset/test.csv", nrows=200)
dataset.drop(['Date','Top1'], axis = 1,inplace=True)
dataset = dataset.values
features = dataset[:,0:4]
label = dataset[:,5:6]
sequence_length = 8

X, y, scaler, scaler1 = prepare_data(features, label, sequence_length)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
dataset = TimeSeriesDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
print(X.shape)


# QINN parameters
input_size = X.shape[2]
hidden_size = 64
output_size = 32

model = QINN(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
best_loss = 10000

if os.path.exists("model/qinn.pt") == False:
    # Training loop
    num_epochs = 1000
    for epoch in range(num_epochs):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}')
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), "model/qinn.pt")
            print("best saved : "+str(best_loss))
    #torch.save(model.state_dict(), "model/gpt.pt")
else:
    model.load_state_dict(torch.load("model/qinn.pt"))
    model.eval()

print(X_test.shape)
print(y_test.shape)
y_test = scaler1.inverse_transform(y_test).ravel()
y_pred = []
for j in range(len(X_test)):
    index = X_test[j]
    temp = []
    temp.append(index)
    index = np.asarray(temp)
    last_sequence = torch.tensor(index, dtype=torch.float32)
    predicted_scaled = model(last_sequence).detach().numpy()
    predicted_scaled = predicted_scaled.reshape(-1, 1)
    predicted_scaled = scaler1.inverse_transform(predicted_scaled)
    predicted_scaled = predicted_scaled.ravel().mean()    
    print(str(y_test[j])+" "+str(predicted_scaled))
    y_pred.append(predicted_scaled)

mse_error = mean_squared_error(y_test, y_pred)
square_error = r2_score(y_test, y_pred)

print(mse_error)
print(square_error)

