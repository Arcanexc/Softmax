import numpy as np
import torch
from torch.utils import data
from Regresion_Lineal import true_w


''' Generamos datos sintéticos '''

def syntetic_data(w, b, num_examples):  # esta función genera datos sintéticos para la regresión lineal
    X = torch.normal(0, 1, (num_examples, len(w)))  # se generan características aleatorias con una distribución normal
    y = torch.matmul(X, w) + b  # se calcula la etiqueta y como el producto punto de X y w más b
    y += torch.normal(0, 0.01, y.shape)  # se añade ruido gaussiano a las etiquetas
    return X, y.reshape((-1, 1))  # se devuelve X y y con la forma adecuada

true_w = torch.tensor([2, -3.4])  # valores verdaderos de los pesos
true_b = 4.2  # valor verdadero del sesgo
features, labels = syntetic_data(true_w, true_b, 1000)  # se generan 1000 ejemplos de datos sintéticos

''' Cargamos los datos en un DataLoader de PyTorch '''

def load_array(data_arrays, batch_size, is_train=True):
    """Crea un DataLoader a partir de los datos proporcionados."""
    dataset = data.TensorDataset(*data_arrays)  # crea un dataset de tensores
    return data.DataLoader(dataset, batch_size, shuffle=is_train)  # devuelve un DataLoader con el tamaño de lote especificado

batch_size = 10  # tamaño del lote
data_iter = load_array((features, labels), batch_size)  # se carga el DataLoader con las características y etiquetas

print(next(iter(data_iter))) # se obtiene el primer lote del DataLoader

''' Definimos el modelo de regresión lineal '''

from torch import nn

net = nn.Sequential(nn.Linear(2, 1))  # se define un modelo secuencial con una capa lineal que toma 2 entradas y produce 1 salida

''' Definimos que es una capa densamente conectada '''

lin = torch.nn.Linear(2, 1)  # se define una capa lineal que toma 2 entradas y produce 1 salida
x = torch.rand(1, 2)  # se crea un tensor de entrada aleatorio con forma (1, 2)
print('Entrada:', x)

print('\n\nPesos y parametros:')
for param in lin.named_parameters():  # se itera sobre los parámetros de la capa lineal
    print(param) # se imprimen los parámetros (pesos y sesgo)

y = lin(x)  # se calcula la salida de la capa lineal
print('\n\nSalida:', y)  # se imprime la salida de la capa lineal

''' Inicializamos los parametros del modelo '''

net[0].weight.data.normal_(0, 0.01)  # se inicializan los pesos de la capa lineal con una distribución normal
net[0].bias.data.fill_(0)  # se inicializa el sesgo de la capa lineal a cero

''' Definimos la función de pérdida '''

loss = nn.MSELoss()  # se define la función de pérdida como el error cuadrático medio

''' Definimos el optimizador '''

import torch.optim as optim

trainer = optim.SGD(net.parameters(), lr=0.03)  # se define el optimizador SGD con una tasa de aprendizaje de 0.03

''' Entrenamos el modelo '''

num_epochs = 8  # número de épocas de entrenamiento
for epoch in range(num_epochs):
    for X, y in data_iter:  # se itera sobre los lotes de datos
        l = loss(net(X), y)  # se calcula la pérdida
        trainer.zero_grad()  # se restablecen los gradientes del optimizador
        l.backward()  # se calcula el gradiente de la pérdida con respecto a los parámetros
        trainer.step()  # se actualizan los parámetros usando el optimizador

    l = loss(net(features), labels)  # se calcula la pérdida en todo el conjunto de entrenamiento
    print(f'epoch {epoch + 1}, loss {l:f}')  # se imprime la pérdida media por época



