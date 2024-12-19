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
from utils.parser import parse_graph_data


def bfs(file_path, start_vertex):
    adjacency_list, adjacency_matrix, num_nodes, headers = parse_graph_data(file_path)

    # Initialisation
    queue = []
    pre_visit = []
    post_visit = []

    queue.append(start_vertex)

    while queue:
        x = queue.pop(0)   # the head of the queue
        pre_visit.append(x)  # mark node

        unmarked_succ = []
        for w in adjacency_list[x]:
            if w[0] not in pre_visit:
                unmarked_succ.append(w[0])

        for y in unmarked_succ:
            pre_visit.append(y)
            queue.append(y)

        post_visit.append(x)

    print(f"Pre-Visit Order: {pre_visit}")
    print(f"Post-Visit Order: {post_visit}")
    print(f"Headers: {headers}")

    return pre_visit, post_visit, headers


if __name__ == "__main__":
    file_path = "../output/bfs-test.json"
    pre_visit, post_visit, headers = bfs(file_path, "H")
    print(f"Pre-Visit Order: {pre_visit}")
    print(f"Post-Visit Order: {post_visit}")
    print(f"Headers: {headers}")
