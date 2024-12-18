import json


def parse_graph_data(file_path):
    """
    Parses graph data from a JSON file and returns the adjacency list, adjacency matrix,
    number of nodes, and headers (node names).

    Parameters:
        file_path (str): Path to the JSON file containing the graph data.

    Returns:
        tuple: A tuple containing:
            - adjacency_list (dict): The graph represented as an adjacency list.
            - adjacency_matrix (list): The graph represented as an adjacency matrix.
            - num_nodes (int): The number of nodes in the graph.
            - headers (list): List of node names (headers).
    """
    # Read JSON content from a file
    with open(file_path, "r") as file:
        json_data = json.load(file)

    # Parse JSON components
    num_nodes = json_data["num_nodes"]
    raw_adjacency_matrix = json_data["adjacency_matrix"]
    raw_adjacency_list = json_data["adjacency_list"]

    # Convert raw adjacency matrix to a proper structure
    adjacency_matrix = []
    headers = raw_adjacency_matrix[0][1:]  # Extract column headers (node names)
    for row in raw_adjacency_matrix[1:]:
        adjacency_matrix.append(row[1:])  # Skip row header during extraction

    # Parse adjacency list into a clean dictionary structure
    adjacency_list = {}
    for node, edges in raw_adjacency_list.items():
        adjacency_list[node] = [(edge["node"], edge["weight"]) for edge in edges]

    return adjacency_list, adjacency_matrix, num_nodes, headers


# Exmple usage (if run as a script):
if __name__ == "__main__":
    file_path = "./output/ape.json"
    adjacency_list, adjacency_matrix, num_nodes, headers = parse_graph_data(file_path)

    print("Number of Nodes:", num_nodes)
    print("\nAdjacency Matrix:")
    for i, row in enumerate(adjacency_matrix):
        print(f"{headers[i]}: {row}")

    print("\nAdjacency List:")
    for node, edges in adjacency_list.items():
        print(f"{node}: {edges}")
