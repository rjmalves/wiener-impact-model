import networkx as nx
from copy import deepcopy
from wiener_calculator import WienerCalculator
from torch_data_reader import TorchDataReader

# Parametros de entrada para o dataset
n = 8
m = 15
DIR = "C:/Users/roger/git/wiener-impact-model/data/test8augmented/raw/"
GRAPH_FILENAME = "graphs.txt"
IMPACT_FILENAME = "impacts.txt"
K = 50  # Quantos grafos 'pai' são desejados


def generate_graph_children(g: nx.Graph):
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

    return graph_g6_dict, wiener_impact_dict


# Enquanto nao tiver gerado K grafos 2-conexos
k = 0
graph_g6_dicts = []
wiener_impact_dicts = []
while k < K:
    # Tenta gerar um grafo aleatorio ate conseguir um 2-conexo
    g = nx.gnm_random_graph(n, m)
    while nx.node_connectivity(g) < 2:
        g = nx.gnm_random_graph(n, m)
    k += 1
    graph_g6_dict, wiener_impact_dict = generate_graph_children(g)
    graph_g6_dicts.append(graph_g6_dict)
    wiener_impact_dicts.append(wiener_impact_dict)

# Escreve os dados num arquivo de texto
with open(DIR+GRAPH_FILENAME, "wb") as f:
    for d in graph_g6_dicts:
        for i in d.keys():
            f.write(d[i])

with open(DIR+IMPACT_FILENAME, "w") as f:
    for d in wiener_impact_dicts:
        for i in d.keys():
            f.write("{}\n".format(d[i]))

datareader = TorchDataReader(DIR)
