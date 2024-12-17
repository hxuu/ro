"""
    ### Principe Général:
        Procéder à un marquage des sommets (utilisation de couleur).
        Chaque sommet passe par 3 états :
        - **Le sommet n’est pas encore visité** ⇒ sommet non marqué (aucune couleur).
        - **Le sommet est atteint pour la première fois et la visite débute** ⇒ sommet ouvert (couleur verte).
        - **Le sommet a été exploré et sa visite est terminée** ⇒ sommet fermé (couleur rouge).

        **L’ordre des sommets :**
        - **Ouverts** ⇒ ordre de prévisite.
        - **Fermés** ⇒ ordre de postvisite.

        - On ouvre successivement tous les successeurs non marqu ́es d’un sommet
        en cours de visite en les ajoutant `a la liste d’ordre de pr ́evisite.
        - L’usage de la structure de donn ́ee FIFO (une file).
"""
import json


def bfs(start_vertex, adjacency_list, adjacency_matrix):
    # Initialize tracking lists
    marked = []  # Vertices that are fully visited (closed)
    non_marked = [row[0] for row in adjacency_matrix[1:]]  # Extract node labels from the matrix

    # Initialize queues
    queue = []  # FIFO queue to track vertices during traversal
    visited = []  # Pre-visit order

    # Start BFS
    queue.append(start_vertex)  # Add the starting vertex to the queue
    marked.append(start_vertex)  # Mark the start vertex as visited
    visited.append(start_vertex)  # Pre-visit order

    while queue:
        # Dequeue the first element
        current = queue.pop(0)

        # Explore all neighbors of the current vertex
        for neighbor in adjacency_list.get(current, []):
            node = neighbor["node"]

            # If the neighbor is not visited
            if node not in marked:
                queue.append(node)  # Add to queue
                marked.append(node)  # Mark as visited
                visited.append(node)  # Add to pre-visit order

    return marked, visited  # Pre-visit order


def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

file_path = '../output/test3.json'
data = load_json(file_path)

# Extract adjacency matrix and list from the loaded JSON data
adjacency_matrix = data["adjacency_matrix"]
adjacency_list = data["adjacency_list"]

# Example usage of the BFS function
start_vertex = "S5"
marked, visited_order = bfs(start_vertex, adjacency_list, adjacency_matrix)
print("\nBFS Pre-visited Order:", visited_order)
print("\nBFS Post-visited Order:", marked)

