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
import pprint
from parser import parse_graph_data
from collections import deque


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


def prufer_decoding(P, sorted_headers):
    adjacency_list = {element: [] for element in sorted_headers}
    # print(adjacency_list)

    while P or len(sorted_headers) > 2:
        # this can probably be more effecient but meh~ Idc
        least_element = [x for x in sorted_headers if x not in P][0]

        # create an edge between least_element and P[0]
        adjacency_list[least_element].append((P[0], None))
        adjacency_list[P[0]].append((least_element, None))

        P.pop(0)
        sorted_headers.remove(least_element)

    adjacency_list[sorted_headers[0]].append((sorted_headers[1], None))
    adjacency_list[sorted_headers[1]].append((sorted_headers[0], None))

    return adjacency_list


if __name__ == "__main__":
    tree_path = "../output/prufer-test.json"  # Path to the tree JSON file
    P, sorted_headers = prufer_encoding(tree_path)
    print("======================================")
    print(f"Prufer Encoding: {P}")
    print(f"Sorted Headers (Nodes): {sorted_headers}")
    adjacency_list = prufer_decoding(P, sorted_headers)
    print("======================================")
    print("Prufer Decoding:")
    pprint.pprint(adjacency_list)

