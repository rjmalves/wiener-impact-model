import networkx as nx
import numpy as np
import scipy as sp
import copy
import time
import itertools
from math import ceil, floor, factorial
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

def reset_graph(vertex_list: list, edge_list: list) -> nx.Graph:
    """
    Cria um grafo novo a partir de uma lista de vértices e uma listsa de arestas.
    """
    graph0 = nx.Graph()
    graph0.add_nodes_from(vertex_list)
    graph0.add_edges_from(edge_list, color = 'black')
        
    return graph0

def print_graph_result(graph_list: List[nx.Graph], initial_graph_complement: nx.Graph):
    """
    Printa uma lista de grafos (com destaques nas arestas adicionadas), dado um grafo original e
    uma lista de novos grafos.
    """
    complement_edges = copy.deepcopy(list(initial_graph_complement.edges()))
    for i in range(len(graph_list)):
        #Creates Aux Graph
        V = copy.deepcopy(list(graph_list[i].nodes()))
        E = copy.deepcopy(list(graph_list[i].edges()))
        aux_initial_graph = reset_graph(V, E)
        #Creates Filter Edges
        filter_edges = [e for e in E if e in complement_edges]
        aux_initial_graph.remove_edges_from(filter_edges)
        aux_initial_graph.add_edges_from(filter_edges, color = 'r')
        edge_list = aux_initial_graph.edges()
        edge_color_list = [aux_initial_graph[x][y]['color'] for x,y in edge_list]
        nx.draw(aux_initial_graph, with_labels=True, node_color='w', edgelist = edge_list, edge_color = edge_color_list)
        plt.show()


def deck_of_graphs(initial_graph: nx.Graph) -> List[nx.Graph]:
    """
    Retorna uma lista de grafos, resultantes da remoção individual de vértices de um grafo base.
    """
    deck = []
    V = copy.deepcopy(list(initial_graph.nodes()))
    E = copy.deepcopy(list(initial_graph.edges()))
    aux_graph = reset_graph(V, E)
    for i in range(len(V)):
        aux_graph.remove_nodes_from([V[i]])
        deck = deck + [aux_graph]
        aux_graph = reset_graph(V, E)
            
    return deck

def deck_of_graphs_edges(initial_graph: nx.Graph) -> List[nx.Graph]:
    """
    Retorna uma lista de grafos, resultantes da remoção individual de arestas de um grafo base.
    """
    deck = []
    V = copy.deepcopy(list(initial_graph.nodes()))
    E = copy.deepcopy(list(initial_graph.edges()))
    aux_graph = reset_graph()
    for i in range(len(E)):
        aux_graph.remove_edges_from([E[i]])
        deck = deck + [aux_graph]        
        aux_graph = reset_graph(V, E)
            
    return deck

def distance_matrix(graph_list: List[nx.Graph]) -> List[np.ndarray]:
    """
    Calcula uma lista de matrizes de distância a partir de uma lista de grafos.
    """
    return [nx.floyd_warshall_numpy(g) for g in graph_list]

def wiener_indices_v_removal(g: nx.Graph) -> List[int]:
    """
    Calcula os índices de Wiener para cada remoção de vértice possível em um grafo.
    """
    deck = deck_of_graphs(g)
    wieners = np.squeeze(np.asarray([nx.wiener_index(x) for x in deck]))
    
    return wieners

def wiener_indices_e_removal(g: nx.Graph) -> List[int]:
    """
    Calcula os índices de Wiener para cada remoção de aresta possível em um grafo.
    """
    deck = deck_of_graphs_edges(g)
    wieners = np.squeeze(np.asarray([nx.wiener_index(x) for x in deck]))
    
    return wieners
    
def wiener_impact_v_removal(g: nx.Graph) -> List[float]:
    """
    Calcula os impactos de Wiener para cada remoção de vértices possível em um grafo.
    """
    n = g.number_of_nodes()
    transmissions = np.squeeze(np.asarray(np.dot(distance_matrix([g]), np.ones(n))))
        
    return wiener_indices_v_removal(g) + transmissions - nx.wiener_index(g) * np.ones(n)

def wiener_impact_e_removal(g: nx.Graph) -> List[float]:
    """
    Calcula os impactos de Wiener para cada remoção de arestas possível em um grafo.
    """
    M = g.number_of_edges()

    return wiener_indices_e_removal(g) - nx.wiener_index(g) * np.ones(M)

def torre(g: nx.Graph, complement_edge_list: List[Tuple[int, int]]) -> List[nx.Graph]:
    """
    Retorna todos os grafos obtidos por adição de arestas a um grafo g.
    """
    M = len(complement_edge_list)
    if M == 0:
        return [g]
    graph_list = []
    V = copy.deepcopy(list(g.nodes()))
    E = copy.deepcopy(list(g.edges()))
    aux_graph = reset_graph(V, E)
    for i in range(M):
        aux_graph.add_edges_from(list(complement_edge_list[i]))
        graph_list.extend([aux_graph])
        aux_graph = reset_graph(V, E)
    
    return graph_list

def graph_to_01(g: nx.Graph) -> List[int]:
    """
    Converte um grafo para a sua representação binária (das arestas possíveis).
    """
    aux_list = list(nx.complete_graph(g.nodes()).edges())
    graph_bin_list = [1 if e in g.edges() else 0 for e in aux_list]
    
    return graph_bin_list

def graph_to_list(graph_list: List[nx.Graph]) -> List[List[int]]:
    """
    Realiza a conversão de grafo para forma binária em uma lista de grafos.
    """
    return [graph_to_01(x) for x in graph_list]

def graph_to_list_differences(g: nx.Graph, original_graph_complement: nx.Graph):
    """
    Converte um grafo para a sua forma binária considerando apenas as arestas
    que foram adicionadas em relação ao seu original.
    """
    complement_edges = list(original_graph_complement.edges())
    graph_edges = list(g.edges())
    bin_differences_list = [1 if e in graph_edges and complement_edges else 0 
                            for e in complement_edges]
#    List_Gr = [0 for e in Complement.edges() if e in Gr.edges()]
    return bin_differences_list


def initial_model_variables(g: nx.Graph) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula a matriz de formas binárias e o vetor de impactos de wiener,
    que são utilizados no treinamento do modelo.
    """
    # Complementar de g
    h = nx.complement(g)
    # Cópia das arestas de h
    aux_edge_list = copy.deepcopy(list(h.edges()))
    # Lista de listas unitárias de arestas de h
    single_edge_lists = list(itertools.combinations(aux_edge_list, 1)) 
    double_edge_lists = list(itertools.combinations(aux_edge_list, 2))
    # Converte para as formas binárias de cada possível adição de aresta
    single_edge_addition_graphs = torre(g, single_edge_lists)
    double_edge_addition_graphs = torre(g, double_edge_lists)
    # Constroi uma matriz com uma forma binária em cada linha.
    # IMPORTANTE: a forma binária considera apenas as diferenças entre as
    #             arestas do grafo original e a do fornecido.
    tmp_list1 = [graph_to_list_differences(graph, h) for graph in single_edge_addition_graphs]
    tmp_list2 = [graph_to_list_differences(graph, h) for graph in double_edge_addition_graphs]
    binary_matrix = np.array(tmp_list1 + tmp_list2)
    # Constroi o vetor com os impactos de wiener de cada adição
    tmp_vec1 = [sum(wiener_impact_v_removal(graph)) for graph in single_edge_addition_graphs]
    tmp_vec2 = [sum(wiener_impact_v_removal(graph)) for graph in double_edge_addition_graphs]
    impacts = np.array(tmp_vec1 + tmp_vec2)

    return binary_matrix, impacts

def calculates_two_additions(g: nx.Graph) -> List[dict]:
    """
    Calcula os impactos gerados, de fato, pela adição de duas arestas e retorna
    a lista dos valores calculados.
    """
    # Complementar de g
    h = nx.complement(g)
    # Cópia das arestas de h
    aux_edge_list = copy.deepcopy(list(h.edges()))
    # Lista de pares de arestas de h
    double_edge_lists = list(itertools.combinations(aux_edge_list, 2))
    # Converte para as formas binárias de cada possível adição de par de arestas
    double_edge_addition_graphs = torre(g, double_edge_lists)
    impacts = np.array([sum(wiener_impact_v_removal(graph)) 
                        for graph in double_edge_addition_graphs])
    actuals = []
    abs_index = 0
    for k in range(len(h.edges())):
        for j in range(k):
            pred_dict = {}
            pred_dict["edge"] = [k, j]
            pred_dict["actual"] = impacts[abs_index]
            actuals.append(pred_dict)
            abs_index += 1

    return actuals


def predicts_two_additions(complement: nx.Graph, model: Ridge) -> List[dict]:
    """
    Calcula as predições para cada adição de duas arestas e retorna a lista predições. 
    Uma predição é um dicionário na forma: {"edge": [u,v], "predicted": y}.
    """
    predictions = []
    # Forma todas as possíveis combinações para adição de arestas.
    # Na verdade, as combinações são tornar 1 os valores da lista binária de 
    #     diferenças, que é fornecida para o modelo.
    complement_edge_num = len(complement.edges())
    for k in range(complement_edge_num):
        for j in range(k):
            pred_dict = {}
            new_vec = np.zeros(complement_edge_num)
            new_vec[j] = 1
            new_vec[k] = 1
            pred_dict["edge"] = [k, j]
            w = model.predict([new_vec])
            pred_dict["predicted"] = w[0]
            predictions.append(pred_dict)
    
    return predictions

def validate_two_additions(g: nx.Graph) -> bool:
    """
    Faz o treinamento de um modelo de predição considerando todas as adições
    de arestas simples e utiliza para prever os impactos de wiener de adições
    de duas arestas.
    """
    # Cria as variáveis que treinam o modelo
    binary_matrix, impacts = initial_model_variables(g)
    # Cria e treina o modelo
    clf = Ridge(alpha = 0.2)
    clf.fit(binary_matrix, impacts)
    # Realiza todas as possíveis adições de duas arestas e obtem as predições
    h = nx.complement(g)
    predictions = predicts_two_additions(h, clf)
    # Obtém os valores de fato dos impactos
    actuals = calculates_two_additions(g)
    complete_list = []
    for i in range(len(predictions)):
        complete_list.append({"actual": actuals[i]["actual"],
                              "predicted": predictions[i]["predicted"]})

    return complete_list

def modela(G):
    H = nx.complement(G)
    #Grafo Complementar de G
    EAux = copy.deepcopy(list(H.edges()))
    #Cópia Profunda das Arestas de H
    A = list(itertools.combinations(EAux, 1))
    #Listas de {e}, onde e é uma aresta de H
    Tur = torre(G, A)
    #Usa A para calcular todos os grafos obtidos de G pela adição de uma única aresta de H
    Matrix = np.array([graph_to_list_differences(Gr, H) for Gr in Tur])
    #Cada linha é a representação binária dos grafos de Tur
    #A representação binária tem 1 se há aresta de H e 0 caso contrário
    vector = np.array([sum(wiener_impact_v_removal(Gr)) for Gr in Tur])
    #Vetor de Impactos de Wiener dos Grafos em Tur
    clf = Ridge(alpha = 0.2)
    clf.fit(Matrix, np.array(vector)) 
    w1 = clf.predict(Matrix)
    print(w1)
    Modelo = clf.coef_
    index_modelo = np.argmin(w1)
    if type(index_modelo) != list:
        index_modelo = [index_modelo]
    edge_list = list(H.edges())
    edge_sol = [edge_list[i] for i in index_modelo]
    #print([edge_list[i] for i in edge_sol])
    print('')
    print('modelo')
    print(Modelo)
    print('aresta')
    print(edge_sol)
    return(edge_sol)