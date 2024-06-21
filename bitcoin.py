import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from dataset.dataloader import data

# 데이터 다운로드
btc = data()
btc.set_index('candle_date_time_kst').inplace=True
# 기술적 지표 계산
def add_indicators(df):
    df['SMA_7'] = df['trade_price'].rolling(window=7).mean()
    df['SMA_30'] = df['trade_price'].rolling(window=24).mean()
    df['RSI'] = calculate_rsi(df['trade_price'], 14)
    df['MACD'], df['Signal'] = calculate_macd(df['trade_price'])
    return df

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, slow=26, fast=12, signal=9):
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal

btc = add_indicators(btc)
btc.dropna(inplace=True)

# 특성 선택
x=btc[[ 'opening_price', 'high_price', 'low_price',
       'trade_price', 'candle_acc_trade_price', 'candle_acc_trade_volume']]
y=btc['target']

# 데이터 정규화
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(x)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

def create_sequences(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 240
X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps)

# 훈련 및 테스트 세트 분할
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# PyTorch 텐서로 변환
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)

# LSTM 모델 정의
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# 모델 초기화
input_dim = X_train.shape[1]
hidden_dim = 50
num_layers = 2
output_dim = 1

model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# 모델 훈련
num_epochs = 50
batch_size = 32

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 예측
model.eval()
with torch.no_grad():
    y_pred = model(X_test).numpy()

# 결과 시각화
plt.figure(figsize=(12, 6))
plt.plot(scaler_y.inverse_transform(y_test), label='Actual Price')
plt.plot(scaler_y.inverse_transform(y_pred), label='Predicted Price')
plt.title('Bitcoin Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# 모델 성능 평가
from sklearn.metrics import mean_squared_error, mean_absolute_error
mse = mean_squared_error(scaler_y.inverse_transform(y_test), scaler_y.inverse_transform(y_pred))
mae = mean_absolute_error(scaler_y.inverse_transform(y_test), scaler_y.inverse_transform(y_pred))
print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')