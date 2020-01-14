import networkx as nx
import matplotlib.pyplot as plt
import os
from copy import deepcopy
from utils.wiener_calculator import WienerCalculator
from utils.torch_data_reader import TorchDataReader

# Parametros de entrada para o dataset
n = 10
m = 25
DIR = "/home/rogerio/git/wiener-impact-model/data/test10nodes/raw/"
GRAPH_FILENAME = "graphs.txt"
IMPACT_FILENAME = "impacts.txt"

# Tenta gerar um grafo aleatorio ate conseguir um 2-conexo
g = nx.gnm_random_graph(n, m)
while nx.node_connectivity(g) < 2:
    g = nx.gnm_random_graph(n, m)

# Inicia as listas de pares: [aresta acrescentada, impacto de wiener].
# O primeiro elemento e [grafo original, 0].
g_comp = nx.complement(g)
edges_in_comp = list(nx.edges(g_comp))

# Cria os dicionarios e processa o grafo original
calc = WienerCalculator(g)
wiener_impact_dict = {}
wiener_impact_dict[-1] = calc.average_edge_wiener_impact
graph_g6_dict = {}
graph_g6_dict[-1] = nx.to_graph6_bytes(g, header=False)

# Processa todos os grafos de adicao de uma unica aresta
for i, e in enumerate(edges_in_comp):
    aux_graph = deepcopy(g)
    aux_graph.add_edges_from([e])
    calc = WienerCalculator(aux_graph)
    wiener_impact_dict[i] = calc.average_edge_wiener_impact
    graph_g6_dict[i] = nx.to_graph6_bytes(aux_graph, header=False)

# Escreve os dados num arquivo de texto
with open(DIR+GRAPH_FILENAME, "wb") as f:
    for i in graph_g6_dict.keys():
        f.write(graph_g6_dict[i])

with open(DIR+IMPACT_FILENAME, "w") as f:
    for i in wiener_impact_dict.keys():
        f.write("{}\n".format(wiener_impact_dict[i]))

datareader = TorchDataReader(DIR)