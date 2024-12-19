"""
    ## Algorithme de Kruskal (1956)

    **Données :**

    * Graphe pondéré G = (V, E) de n sommets et m arêtes,
    * Chaque arête e de E est associée à son poids p(e).

    **Finalité**

    Construire un arbre couvrant de poids minimum A = (V, F).

    **Procédure**

    1. Initialiser F = ∅
    2. Trier et renuméroter les arêtes de G dans l'ordre croissant de leur poids p(e1) ≤ (e2) ≤ ... ≤ (em)
    3. Pour (k = 1, k ≤ m, k + +)
       * si ek ne forme pas de cycle avec F alors F = F ∪ {ek }
"""
from utils.parser import parse_graph_data
from utils.disjoint_set import UnionFind


def kruskal(graph_path):

    # Init
    adjacency_list, adjacency_matrix, num_nodes, headers = parse_graph_data(graph_path)
    edges = []
    for node, successors in adjacency_list.items():
        for succ in successors:
            edges.append((node, succ[0], succ[1]))

    F = []
    sorted_edges = sorted(edges, key=lambda edge: edge[2])

    u = UnionFind(headers)
    min_weight = 0

    for edge in sorted_edges:
        # check if sorted_edges[i] creates a cycle
        if not (u.find(edge[0]) == u.find(edge[1])):
            F.append(edge)
            u.unite(edge[0], edge[1])
            min_weight += edge[2]

    print(f"F: {F}")
    print(f"Minimum Weight: {min_weight}")

    return F, min_weight


if __name__ == "__main__":
    graph_path = "../output/kruskal-test.json"
    F, min_weight = kruskal(graph_path)
    print(f"F: {F}")
    print(f"Minimum Weight: {min_weight}")
