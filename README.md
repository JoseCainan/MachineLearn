# 🎯 Predição de Nota do ENEM com Redes Neurais (PyTorch)

## 📌 Descrição
Este projeto tem como objetivo desenvolver um modelo de **Machine Learning** capaz de prever a nota do ENEM com base em variáveis relacionadas ao desempenho e hábitos de estudo do aluno.

O modelo foi implementado utilizando **PyTorch**, contemplando todas as etapas do pipeline de aprendizado de máquina: preparação dos dados, normalização, treinamento e inferência.

---

## 🧠 Problema

Dado um conjunto de características de um estudante, o modelo busca prever sua nota no ENEM.

### 🔹 Entradas (Features):
- Horas de sono
- Horas de estudo
- Nota de simulado

### 🔹 Saída (Target):
- Nota final do ENEM

---

## 📊 Dataset

O conjunto de dados foi estruturado em formato JSON (`dados_enem.json`) contendo registros como:

```json
{
  "HorasSono": 7,
  "HorasEstudo": 4,
  "NotaSimulado": 750,
  "NotaENEM": 720
}

## 📈 Resultados
O modelo apresentou capacidade de generalização para novos dados, realizando predições consistentes com base nos padrões aprendidos durante o treinamento.
