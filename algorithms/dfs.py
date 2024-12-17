"""
    Principe:
        Utilise aussi le principe de marquage,
        Applicable aux GO et GNO,
        L’usage de la structure de donn ́ee LIFO (une pile).

    Algorithme:
        Initialisation : Tous les sommets sont non marqu ́es.
        Proc ́edure :
        Tant qu’il existe un sommet s non marqu ́e :
            Ouvrir s et l’ins ́erer au sommet de la pile.
        Tant que la pile n’est pas vide faire :
            S’il existe un sommet y non marqu ́e successeur au sommet x situ ́e au
            sommet de la pile.
            Ouvrir y et l’ins ́erer dans la pile.
            Sinon : Fermer x et le supprimer de la pile.
"""
from parser import parse_graph_data


def dfs(file_path):
    parse_graph_data(file_path)
#     # Initialize tracking lists
#     marked = []  # Vertices that are fully visited (closed)
#     non_marked = [row[0] for row in adjacency_matrix[1:]]  # Extract node labels from the matrix
#
#     # Initialize queues
#     queue = []  # FIFO queue to track vertices during traversal
#     visited = []  # Pre-visit order
#
#     # Start BFS
#     queue.append(start_vertex)  # Add the starting vertex to the queue
#     marked.append(start_vertex)  # Mark the start vertex as visited
#     visited.append(start_vertex)  # Pre-visit order
#
#     while queue:
#         # Dequeue the first element
#         current = queue.pop(0)
#
#         # Explore all neighbors of the current vertex
#         for neighbor in adjacency_list.get(current, []):
#             node = neighbor["node"]
#
#             # If the neighbor is not visited
#             if node not in marked:
#                 queue.append(node)  # Add to queue
#                 marked.append(node)  # Mark as visited
#                 visited.append(node)  # Add to pre-visit order
#
#     return marked, visited  # Pre-visit order
#
#

if __name__ == "__main__":
    dfs("../output/ape.json")
