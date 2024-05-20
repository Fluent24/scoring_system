import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.layer1 = torch.nn.Linear(hparams.base_dim, hparams.mlp_hidden) # hidden layer
        self.layer2 = torch.nn.Linear(hparams.mlp_hidden, 1) # output layer
        self.relu = torch.nn.ReLU() # activation function

    def forward(self, x):
        out = self.layer2(self.relu(self.layer1(x)))
        return out

class AudioClassifier1DCNN(nn.Module):
    def __init__(self, input_dim, num_classes=1):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),  # Adjusted input_dim to 1 for CNN
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.linear_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (input_dim // 4), 128),  
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.linear_layers(x)
        return x

class AudioClassifierLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.linear(h_n[-1, :, :])  # Use the last hidden state
        return out

class AudioClassifierTransformer(nn.Module):
    def __init__(self, input_dim, num_classes=1, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128), 
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.embedding(x)  
        x = x.permute(1, 0, 2)  
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  
        x = x.mean(dim=1)  
        x = self.classifier(x)
        return x
