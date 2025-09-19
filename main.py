import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import os
from network_config import get_network_architecture
from network import Network
from visualizer import show_network
from network_config import get_network_architecture
from network import Network


DATASET_PATH = 'heart.csv'
if not os.path.exists(DATASET_PATH):
    print(f"Arquivo {DATASET_PATH} não encontrado. Baixe o dataset do Kaggle e coloque o arquivo na raiz do projeto.")
    exit(1)

# Obter arquitetura da rede através da interface PyQt6
architecture = get_network_architecture()

# Carregar dados
df = pd.read_csv(DATASET_PATH)

# Separar features e target
X = df.drop('target', axis=1)
y = df['target']

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Criar e treinar a rede neural com a arquitetura escolhida
network = Network(architecture)
network.train(X_train, y_train, 
             validation_data=(X_test, y_test),
             epochs=50, 
             batch_size=16)

# Avaliar modelo
preds = (network.predict(X_test) > 0.5).astype(int)
print("\nRelatório de Classificação:")
print(classification_report(y_test, preds))
print(f"Acurácia: {accuracy_score(y_test, preds):.2f}")

# Mostrar visualização da rede neural
show_network(network.model)
