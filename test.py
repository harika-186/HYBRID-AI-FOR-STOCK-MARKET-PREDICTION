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

# --- Time-Series Transformer Model ---
class TimeSeriesTransformer(nn.Module):
    """Time-series Transformer model."""
    def __init__(self, input_size, sequence_length, num_layers, num_heads, d_model, d_ff, dropout):
        super(TimeSeriesTransformer, self).__init__()
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.d_model = d_model

        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = self._get_positional_encoding(sequence_length, d_model)
        self.transformer_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, num_heads, d_ff, dropout),
            num_layers
        )
        self.output_projection = nn.Linear(d_model, 1) # Predict one value

    def _get_positional_encoding(self, seq_len, d_model):
        """Generates positional encodings."""
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        x = self.input_projection(x)
        x = x.permute(1, 0, 2)  # (seq_len, batch, d_model)
        x = x + self.positional_encoding.to(x.device)
        x = self.transformer_layers(x)
        x = x.mean(dim=0)  # Average pooling over sequence length
        x = self.output_projection(x)
        return x.squeeze()

# --- GPT-2 based feature extraction (Example of how to use GPT) ---
def get_gpt_features(text_data):
    """Extracts features from text data using GPT-2."""
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token #prevent warnings

    inputs = tokenizer(text_data, return_tensors='pt', padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    return last_hidden_states.mean(dim=1).detach().numpy() #Average the token embeddings

# --- Example Usage ---
# Assuming you have a pandas DataFrame 'df' with stock price data and 'news_df' with associated news text.
# df = pd.read_csv('stock_data.csv')
# news_df = pd.read_csv('news_data.csv')

# Example data creation for demo. Replace with your actual data.

dataset = pd.read_csv("Dataset/stocks.csv")
dataset.drop(['Ticker','Date', 'Volume'], axis = 1,inplace=True)
dataset = dataset.values
features = dataset[:,0:4]
label = dataset[:,4:5]
num_samples = dataset.shape[0]
sequence_length = 8
news_df = pd.DataFrame({'news': ["Example news " + str(i) for i in range(num_samples)]})

# Example GPT feature extraction.
news_features = get_gpt_features(news_df['news'].tolist()) #Get features from news.
features = np.concatenate((features,news_features), axis=1)
print(features.shape)

X, y, scaler, scaler1 = prepare_data(features, label, sequence_length)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
dataset = TimeSeriesDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
print(X.shape)


# Time-Series Transformer parameters
input_size = X.shape[2]
num_layers = 2
num_heads = 2
d_model = 64
d_ff = 256
dropout = 0.1

model = TimeSeriesTransformer(input_size, sequence_length, num_layers, num_heads, d_model, d_ff, dropout)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(os.path.exists("model/1gpt.pt"))
if os.path.exists("model/gpt.pt") == False:
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}')
    torch.save(model.state_dict(), "model/gpt.pt")
else:
    model.load_state_dict(torch.load("model/gpt.pt"))
    model.eval()

# Prediction example (for the last sequence in your data)
last_sequence = torch.tensor(X_test[-1:], dtype=torch.float32)
predicted_scaled = model(last_sequence).detach().numpy()

predicted_scaled = predicted_scaled.reshape(-1, 1)
predicted_scaled = scaler1.inverse_transform(predicted_scaled)
#predicted_price = scaler.inverse_transform(np.concatenate([predicted_scaled.reshape(1, 1), np.zeros((1, df.shape[1] - 1))], axis=1))[:, 0]
#print(predicted_scaled)

y_test = scaler1.inverse_transform(y_test)
for i in range(0, 10):
    print(str(y_test[i])+" "+str(predicted_scaled[i]))




