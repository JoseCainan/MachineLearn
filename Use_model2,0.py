import torch
from torch import nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.hidden1 = nn.Linear(4, 64)
        self.hidden2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = self.output(x)
        return x

model = NeuralNetwork()
model.load_state_dict(torch.load('modelo_enem_3,0.pth'))
model.eval()

scaler_X = MinMaxScaler()
scaler_X.fit_transform([[6, 2, 650, 0.33]])

scaler_y = MinMaxScaler()
scaler_y.fit_transform([[0], [1000]])  # Ajuste para garantir que as previsões estejam dentro de [0, 1000]

print("Digite os valores para realizar a previsão:")
horas_sono = float(input("Horas de sono (exemplo: 6): "))
horas_estudo = float(input("Horas de estudo (exemplo: 2): "))
nota_simulado = float(input("Nota do simulado (exemplo: 650): "))

razao_estudo_sono = horas_estudo / horas_sono

entrada = np.array([[horas_sono, horas_estudo, nota_simulado, razao_estudo_sono]])

entrada_scaled = scaler_X.transform(entrada)

entrada_tensor = torch.tensor(entrada_scaled, dtype=torch.float32)

with torch.no_grad():
    pred_scaled = model(entrada_tensor).squeeze().numpy()

pred_clamped = np.clip(pred_scaled, 0, 1)  # Garantir que os valores normalizados fiquem no intervalo [0, 1]

pred_original = scaler_y.inverse_transform(pred_clamped.reshape(-1, 1))

print(f"\nPrevisão da nota do ENEM: {pred_original[0][0]:.2f}")


