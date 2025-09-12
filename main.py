
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from tensorflow import keras
import os
import tkinter as tk
from tkinter import simpledialog, messagebox


DATASET_PATH = 'heart.csv'
if not os.path.exists(DATASET_PATH):
    print(f"Arquivo {DATASET_PATH} não encontrado. Baixe o dataset do Kaggle e coloque o arquivo na raiz do projeto.")
    exit(1)

# Interface para escolher número de neurônios
def get_neuron_settings():
    root = tk.Tk()
    root.withdraw()
    n1 = simpledialog.askinteger("Configuração da Rede Neural", "Número de neurônios na 1ª camada oculta:", initialvalue=16, minvalue=1, maxvalue=128)
    n2 = simpledialog.askinteger("Configuração da Rede Neural", "Número de neurônios na 2ª camada oculta:", initialvalue=8, minvalue=1, maxvalue=128)
    if n1 is None or n2 is None:
        messagebox.showinfo("Cancelado", "Execução cancelada pelo usuário.")
        exit(0)
    return n1, n2

n_neurons_1, n_neurons_2 = get_neuron_settings()

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

# Construir modelo com parâmetros definidos pelo usuário
model = keras.Sequential([
    keras.layers.Dense(n_neurons_1, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(n_neurons_2, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinar modelo
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.1, verbose=1)

# Avaliar modelo
preds = (model.predict(X_test) > 0.5).astype(int)
print("\nRelatório de Classificação:")
print(classification_report(y_test, preds))
print(f"Acurácia: {accuracy_score(y_test, preds):.2f}")
