import networkx as nx
import numpy as np
import copy
from typing import List, Dict, Tuple

class WienerCalculator:

    def __init__(self, graph: nx.Graph):
        self.__graph = graph
        self.__nodes = list(graph.nodes())
        self.__n = len(self.__nodes)
        self.__edges = list(graph.edges())
        self.__m = len(self.__edges)
        self.__distance_matrix = nx.floyd_warshall_numpy(graph)
        self.__transmissions = self.__transmissions_init()
        self.__wiener_index = self.__distance_matrix.sum() / 2

        self.__node_removals_graphs = self.__node_removal_graphs_init()
        self.__edge_removals_graphs = self.__edge_removal_graphs_init()

        self.__node_removal_wieners = self.__node_removal_wieners_init()
        self.__edge_removal_wieners = self.__edge_removal_wieners_init()

        self.__node_wiener_impacts = self.__node_wiener_impact_init()
        self.__edge_wiener_impacts = self.__edge_wiener_impact_init()

    def __transmissions_init(self) -> Dict[int, float]:
        transmissions_dict = {}
        for i,v in enumerate(self.__nodes):
            transmissions_dict[v] = self.__distance_matrix[i,:].sum(axis=1).item()
        return transmissions_dict

    def __node_removal_graphs_init(self) -> Dict[int, nx.Graph]:
        """
        Retorna uma lista de grafos, resultantes da remoção individual de vértices de um grafo base.
        """
        node_removal_dict = {}
        aux_graph = copy.deepcopy(self.__graph)
        for v in self.__nodes:
            aux_graph.remove_nodes_from([v])
            node_removal_dict[v] = aux_graph
            aux_graph = copy.deepcopy(self.__graph)
                
        return node_removal_dict

    def __edge_removal_graphs_init(self) -> Dict[Tuple[int, int], nx.Graph]:
        """
        Retorna uma lista de grafos, resultantes da remoção individual de arestas de um grafo base.
        """
        edge_removal_dict = {}
        aux_graph = copy.deepcopy(self.__graph)
        for e in self.__edges:
            aux_graph.remove_edges_from([e])
            edge_removal_dict[e] = aux_graph
            aux_graph = copy.deepcopy(self.__graph)
                
        return edge_removal_dict

    def __node_removal_wieners_init(self) -> Dict[int, float]:
        wiener_dict = {}
        for v, g in self.__node_removals_graphs.items():
            wiener_dict[v] = nx.wiener_index(g)
        return wiener_dict
    
    def __edge_removal_wieners_init(self) -> Dict[Tuple[int, int], float]:
        wiener_dict = {}
        for e, g in self.__edge_removals_graphs.items():
            wiener_dict[e] = nx.wiener_index(g)
        return wiener_dict

    def __node_wiener_impact_init(self) -> List[float]:
        impacts = []
        for v in self.__nodes:
            imp = self.__node_removal_wieners[v] - self.__wiener_index + self.__transmissions[v]
            impacts.append([imp])

        return impacts

    def __edge_wiener_impact_init(self) -> float:
        impacts = []
        for e in self.__edges:
            impacts.append(self.__edge_removal_wieners[e] - self.__wiener_index)
            
        return impacts

    @property
    def node_wiener_impacts(self):
        return self.__node_wiener_impacts

    @property
    def average_node_wiener_impact(self):
        return sum(self.__node_wiener_impacts) / self.__n
    
    @property 
    def average_edge_wiener_impact(self):
        return sum(self.__edge_wiener_impacts) / self.__m