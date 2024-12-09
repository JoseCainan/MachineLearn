import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import json

with open('dadosenem.json', 'r') as file:
    data = json.load(file)

df = pd.DataFrame(data)
print(f"Dados carregados. Tamanho do dataset: {df.shape}")

X = df[['HorasSono', 'HorasEstudo', 'NotaSimulado']].values
y = df['NotaENEM'].values
print(f"Colunas de entrada (X): {X[:5]}")
print(f"Coluna de saída (y): {y[:5]}")

scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
print(f"Dados normalizados. Primeiras entradas normalizadas: {X_scaled[:5]}")
print(f"Saídas normalizadas: {y_scaled[:5]}")

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
print(f"Tamanhos do conjunto de treino: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Tamanhos do conjunto de teste: X_test={X_test.shape}, y_test={y_test.shape}")

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.hidden1 = nn.Linear(X_train.shape[1], 64)
        self.hidden2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = self.output(x)
        return x

model = NeuralNetwork()
print(f"Modelo criado: \n{model}")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 50
patience = 10
best_loss = float('inf')
patience_counter = 0

print("\nIniciando treinamento...")
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    predictions = model(X_train_tensor).squeeze()
    loss = criterion(predictions, y_train_tensor)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_predictions = model(X_test_tensor).squeeze()
        val_loss = criterion(val_predictions, y_test_tensor)

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

    if val_loss < best_loss:
        print(f"Melhoria na validação! Salvando modelo com Val Loss: {val_loss.item():.4f}")
        best_loss = val_loss
        best_model = model.state_dict()
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"Sem melhoria ({patience_counter}/{patience})")
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

model.load_state_dict(best_model)

model.eval()
with torch.no_grad():
    test_predictions = model(X_test_tensor).squeeze()
    test_loss = criterion(test_predictions, y_test_tensor)
print(f"\nMean Squared Error (MSE) no conjunto de teste: {test_loss.item():.2f}")

y_pred_scaled = test_predictions.numpy()
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))

print("\nResultados:")
print("Previsões (Primeiros 5 valores):")
print(y_pred[:5])
print("Valores reais (Primeiros 5 valores):")
print(scaler_y.inverse_transform(y_test[:5]))

model.load_state_dict(best_model)

torch.save(model.state_dict(), 'modelo_enem.pth')
print("Modelo treinado salvo em 'modelo_enem.pth'")
