import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import json

# Carregar os dados
with open('dados_enem.json', 'r') as file:
    data = json.load(file)

df = pd.DataFrame(data)
print(f"Dados carregados. Tamanho do dataset: {df.shape}")

# Adicionar a coluna Razão Estudo/Sono
df['RazaoEstudoSono'] = df['HorasEstudo'] / df['HorasSono']
print(f"Primeiras entradas com a nova coluna RazaoEstudoSono:\n{df[['HorasEstudo', 'HorasSono', 'RazaoEstudoSono']].head()}")

# Separar dados de entrada e saída
X = df[['HorasSono', 'HorasEstudo', 'NotaSimulado', 'RazaoEstudoSono']].values
y = df['NotaENEM'].values

# Normalizar os dados
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# Divisão em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Conversão para tensores
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).squeeze()
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).squeeze()

# Definir a classe da Rede Neural
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

# Criar o modelo
model = NeuralNetwork()
print(f"Modelo criado: \n{model}")

# Configurar função de perda e otimizador
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Parâmetros de treinamento
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

    # Avaliação no conjunto de teste
    model.eval()
    with torch.no_grad():
        val_predictions = model(X_test_tensor).squeeze()
        val_loss = criterion(val_predictions, y_test_tensor)

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

    # Early Stopping
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

# Carregar os melhores pesos
model.load_state_dict(best_model)

# Avaliação final
model.eval()
with torch.no_grad():
    test_predictions = model(X_test_tensor).squeeze()
    test_loss = criterion(test_predictions, y_test_tensor)

# Aplicar clamp para garantir previsões no intervalo esperado
test_predictions_clamped = torch.clamp(test_predictions, min=0, max=1)
print(f"\nMean Squared Error (MSE) no conjunto de teste: {test_loss.item():.2f}")

# Reverter a escala para as previsões
y_pred_scaled = test_predictions_clamped.numpy()
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))

# Exibir os resultados
print("\nResultados:")
print("Previsões (Primeiros 5 valores):")
print(y_pred[:5])
print("Valores reais (Primeiros 5 valores):")
print(scaler_y.inverse_transform(y_test[:5].reshape(-1, 1)))

# Salvar o modelo
torch.save(model.state_dict(), 'modelo_enem_3,0.pth')
print("Modelo treinado salvo em 'modelo_enem_3,0.pth'")
