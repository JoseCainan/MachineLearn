import torch
from torch import nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Recriar a classe da rede neural
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.hidden1 = nn.Linear(3, 64)  # Ajuste o número de entradas (3 no caso)
        self.hidden2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = self.output(x)
        return x

# Inicializar o modelo e carregar os pesos
model = NeuralNetwork()
model.load_state_dict(torch.load('modelo_enem.pth'))
model.eval()  # Colocar o modelo em modo de avaliação

# Simular os scalers usados no treinamento (ajustar ao conjunto original usado no treinamento)
scaler_X = MinMaxScaler()
scaler_X.fit_transform([[6, 2, 650]])  # Ajustar aos valores do treino original

scaler_y = MinMaxScaler()
scaler_y.fit_transform([[300], [900]])  # Ajustar ao intervalo da saída no treinamento

# Solicitar ao usuário os valores de entrada
print("Digite os valores para realizar a previsão:")
horas_sono = float(input("Horas de sono (exemplo: 6): "))
horas_estudo = float(input("Horas de estudo (exemplo: 2): "))
nota_simulado = float(input("Nota do simulado (exemplo: 650): "))

# Criar a entrada baseada no input do usuário
entrada = np.array([[horas_sono, horas_estudo, nota_simulado]])

# Normalizar os dados de entrada
entrada_scaled = scaler_X.transform(entrada)

# Converter para tensor PyTorch
entrada_tensor = torch.tensor(entrada_scaled, dtype=torch.float32)

# Fazer a previsão
with torch.no_grad():
    pred_scaled = model(entrada_tensor).squeeze().numpy()

# Reverter a escala para o valor original
pred_original = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))

# Exibir o resultado
print(f"\nPrevisão da nota do ENEM: {pred_original[0][0]:.2f}")
