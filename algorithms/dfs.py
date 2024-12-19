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
from utils.parser import parse_graph_data


def dfs(file_path, start_vertex):
    adjacency_list, adjacency_matrix, num_nodes, headers = parse_graph_data(file_path)

    # Initialisation
    stack = []
    pre_visit = []
    post_visit = []

    stack.append(start_vertex)
    i = 0
    while stack:
        x = stack[-1]   # peek at the top of the stack
        if x not in pre_visit:
            pre_visit.append(x)

        unmarked_succ = []
        for w in adjacency_list[x]:
            if w[0] not in pre_visit:
                unmarked_succ.append(w[0])
        if unmarked_succ:   # there exist an unmarked successor
            y = unmarked_succ[0]
            stack.append(y)
        else:
            post_visit.append(x)
            stack.pop()

    print(f"Pre-Visit Order: {pre_visit}")
    print(f"Post-Visit Order: {post_visit}")
    print(f"Headers: {headers}")

    return pre_visit, post_visit, headers


if __name__ == "__main__":
    file_path = "../output/dfs-test.json"
    pre_visit, post_visit, headers = dfs(file_path, "A")
    print(f"Pre-Visit Order: {pre_visit}")
    print(f"Post-Visit Order: {post_visit}")
    print(f"Headers: {headers}")
