from functions import *
import networkx as nx 

def main():
    N = 8
    S = 15
    all_errors = []
    for __ in range(100):
        G0 = nx.gnm_random_graph(N,S)
        if nx.node_connectivity(G0) >= 2:
            l = validate_two_additions(G0)
            actuals = [i["actual"] for i in l]
            predicted = [i["predicted"] for i in l]
            errors = [actual - predicted for actual, predicted in zip(actuals, predicted)]
            all_errors += errors
    plt.hist(all_errors)
    plt.show()


if __name__ == "__main__":
    main()