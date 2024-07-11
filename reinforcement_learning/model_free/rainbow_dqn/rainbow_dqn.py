import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LinearDataset(Dataset):
    def __init__(self, m, b, num_samples=1000):
        self.x = torch.linspace(-10, 10, num_samples).unsqueeze(1)
        self.y = m * self.x + b

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc_in = nn.Linear(input_dim, model_dim)
        self.fc_out = nn.Linear(model_dim, output_dim)

    def forward(self, src):
        src = self.fc_in(src)
        src = self.transformer_encoder(src)
        output = self.fc_out(src)
        return output


def train_model(model, dataloader, num_epochs=50, learning_rate=0.001, device='cpu'):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(x_batch.unsqueeze(-1))
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(dataloader):.4f}")

    return model

def predict(model, x_sequence, device='cpu'):
    model.eval()
    with torch.no_grad():
        x_sequence = torch.tensor(x_sequence).float().unsqueeze(-1).to(device)
        predictions = model(x_sequence).squeeze(-1).cpu()
    return predictions.numpy()


def train_and_return_model(m, b, num_samples=1000, num_epochs=50, learning_rate=0.001, device='cpu'):
    # Create dataset and dataloader
    dataset = LinearDataset(m, b, num_samples)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Define the model
    input_dim = 1
    model_dim = 64
    num_heads = 4
    num_layers = 2
    output_dim = 1
    model = TransformerModel(input_dim, model_dim, num_heads, num_layers, output_dim)

    # Train the model
    trained_model = train_model(model, dataloader, num_epochs, learning_rate, device)

    return trained_model

def predict_with_trained_model(trained_model, x_sequence):
    return predict(trained_model, x_sequence)



