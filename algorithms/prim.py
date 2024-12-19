"""
Algorithme de Jarnik (1930), Prim (1957)

Entrées :
    Graphe connexe G(V, E) avec une valuation positive des arêtes.

Sortie :
    Un arbre couvrant de poids minimum.
    F : ensemble des arêtes de l'arbre.
    U : ensemble des sommets connectés par F.

Étapes :
    1. Initialiser F = ∅.
    2. Choisir arbitrairement un sommet s.
    3. Initialiser U = {s}.
    4. Tant que U ≠ V :
        - Sélectionner l'arête (s_i, s_j) du cocycle de U de poids minimum.
        - Ajouter l'arête à F : F = F ∪ {(s_i, s_j)}.
        - Ajouter le sommet s_j à U : U = U ∪ {s_j}.
    5. Fin.
"""
from utils.parser import parse_graph_data


def co(U, adjacency_list):
    cocycle = []
    for v in U:
        for adj in adjacency_list[v]:
            # print(adj[0])
            if adj[0] not in U:
                # means the edge added doesn't form a cycle
                cocycle.append((v, adj[0], adj[1]))

    return sorted(cocycle, key=lambda x: x[2])


def prim(graph_path):

    # Init
    adjacency_list, adjacency_matrix, num_nodes, headers = parse_graph_data(graph_path)
    edges = []
    for node, successors in adjacency_list.items():
        for succ in successors:
            edges.append((node, succ[0], succ[1]))

    # random_header = random.choice(headers)
    F = []
    U = {"S1"}
    min_weight = 0

    while U != set(headers):
        cocyle_U = co(U, adjacency_list)

        F.append(cocyle_U[0])  # minimum weight already
        U.add(cocyle_U[0][1])

        min_weight += cocyle_U[0][2]

    return F, min_weight


if __name__ == "__main__":
    graph_path = "../output/kruskal-test.json"
    F, min_weight = prim(graph_path)
    print(f"F: {F}")
    print(f"Minimum Weight: {min_weight}")

