def extract_graph_json(response, file_path):
    """
    Extracts the graph JSON from the response and saves it to the specified file.

    Parameters:
        response: The API response containing the JSON.
        file_path: The path to save the extracted JSON file.
    """
    # Access the candidates from the response object
    candidates = response._result.candidates

    if not candidates:
        raise ValueError("No candidates found in the response.")

    # Extract the text content from the first candidate
    json_candidate = candidates[0].content.parts[0].text

    # Extract the JSON string between the backticks
    json_start = json_candidate.find('```') + 3
    json_end = json_candidate.rfind('```')
    graph_json_str = json_candidate[json_start:json_end].strip()

    # Debugging: print the extracted JSON string
    # print(f"Extracted JSON string: {graph_json_str}")

    if not graph_json_str:
        raise ValueError("Extracted JSON string is empty.")

    # Remove the first line if it is `= json\n`
    graph_json_lines = graph_json_str.splitlines()
    if graph_json_lines and graph_json_lines[0] == "json":
        graph_json_str = "\n".join(graph_json_lines[1:])

    # Save the JSON string to the file
    with open(file_path, 'w') as f:
        f.write(graph_json_str)

