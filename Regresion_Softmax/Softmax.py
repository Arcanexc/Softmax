import torch
import torchvision
from torch import nn
from IPython import display
from torchvision import transforms
from torch.utils import data


def load_data_fashion_mnist(batch_size, resize=None):  # esta función carga el dataset FashionMNIST
    """Download the Fashion-MNIST dataset and return data iterators."""
    trans = [transforms.ToTensor()]  # transforms.ToTensor() convierte las imágenes a tensores
    if resize:
        trans.insert(0, transforms.Resize(resize))  # si se especifica un tamaño de redimensionamiento, se añade al principio de la lista
    trans = transforms.Compose(trans)                       # se combinan las transformaciones en una sola
    mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download=True)  # se descarga el dataset de entrenamiento
    mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans, download=True)   # se descarga el dataset de prueba
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,    # se crea un DataLoader para el dataset de entrenamiento
                            num_workers=1),
            data.DataLoader(mnist_test, batch_size, shuffle=False,    # se crea un DataLoader para el dataset de prueba
                            num_workers=1))


if __name__ == "__main__":   # este bloque se ejecuta si el script se ejecuta directamente
    batch_size = 256   # tamaño del batch para el DataLoader
    train_iter, test_iter = load_data_fashion_mnist(batch_size)   # se cargan los DataLoaders para el dataset de entrenamiento y prueba

    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

    def init_weights(m):  # esta función inicializa los pesos de la red
        if type(m) == nn.Linear:  # si el módulo es una capa lineal
            nn.init.normal_(m.weight, std=0.01)  # se inicializan los pesos con una distribución normal con desviación estándar 0.01

    net.apply(init_weights)

    loss = nn.CrossEntropyLoss(reduction='none')
    trainer = torch.optim.SGD(net.parameters(), lr=0.1)

    def accuracy(y_hat, y):
        """Compute the number of correct predictions."""
        if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
            y_hat = y_hat.argmax(axis=1)
        cmp = y_hat.type(y.dtype) == y
        return float(cmp.type(y.dtype).sum())

    num_epochs = 10
    lr = 0.01
    for epoch in range(num_epochs):
        L = 0.0
        N = 0
        Acc = 0.0
        TestAcc = 0.0
        TestN = 0
        for X, y in train_iter:
            l = loss(net(X) ,y)
            trainer.zero_grad()
            l.mean().backward()
            trainer.step()
            L += l.sum()
            N += l.numel()
            Acc += accuracy(net(X), y)
        for X, y in train_iter:
            TestN += y.numel()
            TestAcc += accuracy(net(X), y)
        print(f'epoch {epoch + 1}, loss {(L/N):f}\
              , train accuracy  {(Acc/N):f}, test accuracy {(TestAcc/TestN):f}')
