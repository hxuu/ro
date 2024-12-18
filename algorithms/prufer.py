"""
Algorithme :

- Données :
     Arbre A.
- Répéter
     Identifier la feuille s; de l'arbre ayant le numéro minimum,
     Ajouter à la suite P le seul sommet s; adjacent à s; dans l'arbre A,
     Enlever de l'arbre A le sommet s; et l'arête incidente à s¡,
- Critère d'arrêt :
     Il ne reste que deux sommets dans l'arbre A.
"""
from parser import parse_graph_data
from collections import defaultdict, deque

def prufer_encoding(tree_path):
    adjacency_list, _, _, headers = parse_graph_data(tree_path)

    # Init
    P = []

    # Create a deque for nodes with only one neighbor (leaf nodes)
    leaf_nodes = deque(sorted([node for node, neighbors in adjacency_list.items() if len(neighbors) == 1]))

    while len(adjacency_list) > 2:
        leaf = leaf_nodes.popleft()

        neighbor = adjacency_list[leaf][0][0]  # each neighbor has (node, weight)

        P.append(neighbor)

        # Remove the leaf node from the adjacency list and from its neighbor's list
        adjacency_list[neighbor] = [n for n in adjacency_list[neighbor] if n[0] != leaf]
        del adjacency_list[leaf]

        # If the neighbor becomes a leaf, add it to the leaf_nodes deque
        if len(adjacency_list[neighbor]) == 1:
            leaf_nodes.append(neighbor)
            leaf_nodes = deque(sorted(leaf_nodes))

    return P, sorted(headers)


if __name__ == "__main__":
    tree_path = "../output/prufer-test.json"  # Path to the tree JSON file
    P, headers_sorted = prufer_encoding(tree_path)
    print(f"Prufer Encoding: {P}")
    print(f"Sorted Headers (Nodes): {headers_sorted}")

