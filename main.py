# from data.test8nodes.dataset import Test8Nodes
# from data.test10nodes.dataset import Test10Nodes
from data.test8augmented.dataset import Test8Augmented
from torch_geometric.data import DataLoader
import torch.nn.functional as F
from models.net import Net
import torch
import matplotlib.pyplot as plt


def main():
    DIR = "C:/Users/roger/git/wiener-impact-model/data/test8augmented"
    net = Net()

    train_dataset = Test8Augmented(root=DIR, name="Test8Augmented")
    train_loader = DataLoader(train_dataset, batch_size=5)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=5e-4)

    # Treinamento
    losses = []
    for epoch in range(5):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = net(data)
            loss = F.mse_loss(outputs, data.y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        losses.append(running_loss)
        # print("Epoch: {} - Running Loss: {}".format(epoch + 1, running_loss))
    # Teste
    erro = []
    predicted = []
    target = []
    for data in train_loader:
        for e in net(data):
            predicted.append(e.item())
        for e in data.y:
            target.append(e.item())
        for e, i in zip(net(data), data.y):
            erro.append(F.mse_loss(e, i).item())
    print("MSE: {}".format(sum(erro) / len(erro)))

    plt.subplot(2, 1, 1)
    plt.plot(losses)
    plt.subplot(2, 1, 2)
    plt.plot(predicted)
    plt.plot(target)
    plt.show()


if __name__ == "__main__":
    main()
