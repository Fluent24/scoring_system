# Copyright (c) 2022 kakaoenterprise
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# 1DCNN: 1차원 합성곱을 사용하여 음성 데이터의 지역적인 특징을 추출합니다. MaxPooling으로 특징 맵의 크기를 줄이고, Flatten하여 완전 연결 계층에 전달합니다.
# RNN (LSTM): 시퀀스 데이터의 시간적인 의존성을 모델링합니다. LSTM은 RNN의 변형으로, 장기 의존성을 학습하는 데 효과적입니다. 마지막 hidden state를 사용하여 분류합니다.
# Transformer: Self-attention 메커니즘을 사용하여 입력 시퀀스의 모든 위치 간 관계를 모델링합니다. 병렬 처리가 가능하고 장기 의존성 모델링에 뛰어나지만, 계산량이 많을 수 있습니다.

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
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1), 
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
        out = self.linear(h_n[-1, :, :])  # 마지막 hidden state 사용
        return out
    

class AudioClassifierTransformer(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, dim_feedforward, num_classes=1):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        out = self.transformer_encoder(x)
        out = out.mean(dim=1)  # sequence dimension에 대해 평균
        out = self.linear(out)
        return out