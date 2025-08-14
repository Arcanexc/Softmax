#%matplotlib inline
import random
import torch
from matplotlib import pyplot as plt

''' Regresion Lineal con python puro y PyTorch '''

def synthetic_data(w, b, num_examples):  #   esta función genera datos sintéticos para la regresión lineal recibe un vector de pesos w, un sesgo b y el número de ejemplos num_examples
    X = torch.normal(0, 1, (num_examples, len(w)))        # X es una matriz de tamaño num_examples x len(w) con valores normales
    y = torch.matmul(X, w) + b       # y es el producto punto de X y w más b
    y += torch.normal(0, 0.01, y.shape)  # se añade ruido a y o error aleatorio con una desviación estándar de 0.01
    return X, y.reshape((-1, 1))     # se devuelve X y y como tensores de PyTorch

'''valores verdaderos de los pesos y el sesgo'''
true_w = torch.tensor([2, -3.4])         # true_w es un tensor que representa los pesos verdaderos
true_b = 4.2                   # true_b es un tensor que representa el sesgo verdadero
features, labels = synthetic_data(true_w, true_b, 1000)  # se generan 1000 ejemplos de datos sintéticos

print('features:', features.shape,'\nlabel:', labels.shape) # se imprimen las formas de los tensores features y labels

plt.rcParams['figure.figsize'] = (4.0, 3.5)  # se establece el tamaño de la figura
# Punto y coma para mostrar solo la figura
plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1)  # se crea un gráfico de dispersión de los datos sintéticos
#plt.show()  # se muestra el gráfico

''' Cargamos los datos en un DataLoader de PyTorch '''

def data_iter(batch_size, features, labels):  # esta función devuelve un iterador de datos que devuelve lotes de tamaño batch_size, donde batch_size es el tamaño del lote, features son las características y labels son las etiquetas
    num_examples = len(features)  # se obtiene el número de ejemplos
    indices = list(range(num_examples))  # se crea una lista de índices
    # aleotorizamos el orden de los datos
    random.shuffle(indices)  # se mezclan los índices aleatoriamente
    for i in range(0, num_examples, batch_size):  # se itera sobre los índices en pasos de batch_size
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num_examples)]) # esta línea crea un tensor de índices de lote
        yield features[batch_indices], labels[batch_indices]  # se devuelve un lote de características y etiquetas



batch_size = 10  # se establece el tamaño del lote
for X, y in data_iter(batch_size, features, labels):  # se itera sobre los lotes de datos
    print(X, '\n', y)  # se imprimen las características y etiquetas del lote
    break  # se rompe el bucle después del primer lote


''' Valores iniciales de nuestro modelo '''

w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)  # se inicializan los pesos w con valores normales pequeños y se establece requires_grad=True para calcular los gradientes
b = torch.zeros(1, requires_grad=True)  # se inicializa el sesgo b en cero y se establece requires_grad=True para calcular el gradiente

''' Definimos el modelo de regresión lineal '''

def linreg(X, w, b):  # esta función define el modelo de regresión lineal
    return torch.matmul(X, w) + b  # se devuelve el producto punto de X y w más b

''' Definimos la función de pérdida '''

def squared_loss(y_hat, y):  # esta función define la pérdida cuadrática
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2  # se devuelve la pérdida cuadrática media

''' Definimos el optimizador '''

def sgd(params, lr, batch_size):  # esta función define el optimizador de descenso de gradiente estocástico (SGD) donde params son los parámetros del modelo, lr es la tasa de aprendizaje y batch_size es el tamaño del lote
    """Actualiza los parámetros en una sola iteración de SGD."""
    with torch.no_grad():  # se desactiva el cálculo de gradientes
        for param in params:  # se itera sobre los parámetros
            param -= lr * param.grad / batch_size  # se actualizan los parámetros restando el gradiente escalado por la tasa de aprendizaje y el tamaño del lote
            param.grad.zero_()  # se restablecen los gradientes a cero

''' Entrenamos el modelo '''

lr = 0.03  # se establece la tasa de aprendizaje
num_epochs = 8  # se establece el número de épocas
net = linreg  # se define la red como la función de regresión lineal
loss = squared_loss  # se define la función de pérdida como la pérdida cuadrática



for epoch in range(num_epochs):  # se itera sobre el número de épocas
    for X, y in data_iter(batch_size, features, labels):  # se itera sobre los lotes de datos
        l = loss(net(X, w, b), y)  # se calcula la pérdida
        l.sum().backward()  # se calcula el gradiente de la pérdida con respecto a los parámetros
        sgd([w, b], lr, batch_size)  # se actualizan los parámetros usando el optimizador SGD
    with torch.no_grad():  # se desactiva el cálculo de gradientes
        train_l = loss(net(features, w, b), labels)  # se calcula la pérdida en todo el conjunto de entrenamiento
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')  # se imprime la pérdida media por época


print(f'error de estimación en w: {true_w - w.reshape(true_w.shape)}')  # se imprime el error de estimación en los pesos
print(f'error de estimación en b: {true_b - b}')  # se imprime el error de estimación en el sesgo
plt.rcParams['figure.figsize'] = (4.0, 3.5)  # se establece el tamaño de la figura
plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1)
plt.scatter(features[:, (1)].detach().numpy(), net(features, w, b).detach().numpy(), 1, color='b')  # se crea un gráfico de dispersión de los datos sintéticos y la línea de regresión
#plt.show()  # se muestra el gráfico