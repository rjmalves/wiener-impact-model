from data.test8nodes.dataset import Test8Nodes
from data.test10nodes.dataset import Test10Nodes
from torch_geometric.data import DataLoader
import torch.nn.functional as F
from models.net import Net
from torch_geometric.utils import to_networkx
import torch
import matplotlib.pyplot as plt


def main():
    DIR = "/home/rogerio/git/wiener-impact-model/data/test10nodes"
    net = Net()

    train_dataset = Test10Nodes(root=DIR, name="Test10Nodes")
    train_loader = DataLoader(train_dataset, batch_size=1)
    for data in train_loader:
        d = data.to_data_list()[0]
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=5e-4)

    # Treinamento
    losses = []
    for epoch in range(20):
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
        predicted.append(net(data).item())
        target.append(data.y.item())
        erro.append(F.mse_loss(net(data), data.y).item())
    print("MSE: {}".format(sum(erro) / len(erro)))

    plt.plot(losses)
    plt.show()
    
if __name__ == "__main__":
    main()