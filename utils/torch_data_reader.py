import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import networkx as nx
import numpy as np
from typing import List
from utils.wiener_calculator import WienerCalculator

class TorchDataReader:

    def __init__(self, data_dir: str):
        self.__graphs = self.__graphs_read(data_dir)
        self.__impacts = self.__impacts_read(data_dir)
        self.__torch_data = self.__torch_data_init()

    def __graphs_read(self, data_dir: str) -> List[nx.DiGraph]:
        graphs = []
        with open(data_dir + "graphs.txt", "rb") as f:
            for g in f:
                graphs.append(nx.from_graph6_bytes(g[:-1]).to_directed())
        return graphs

    def __impacts_read(self, data_dir: str)-> List[float]:
        impacts = []
        with open(data_dir + "impacts.txt", "r") as f:
            for i in f:
                impacts.append(float(i))
        return impacts

    def __torch_data_init(self) -> List[Data]:
        data = []
        for g, i in zip(self.__graphs, self.__impacts):
            # Cria um vetor de atributos default
            n = len(g.nodes())
            calc = WienerCalculator(g)
            impacts = calc.node_wiener_impacts
            x = torch.tensor(impacts, dtype=torch.float)
            # Cria uma lista de arestas e converte para tensor
            edge_list = [[e[0], e[1]] for e in list(g.edges())]
            edge_index = torch.tensor(edge_list, dtype=torch.long)
            # Define o impacto como atributo alvo do grafo
            y = torch.tensor([[i]], dtype=torch.float)
            # Cria o objeto Data e adiciona a lista
            data.append(Data(x=x, edge_index=edge_index.t().contiguous(), y=y))
        return data

    @property
    def graphs(self):
        return self.__graphs
    
    @property
    def impacts(self):
        return self.__impacts

    @property
    def torch_data(self):
        return self.__torch_data