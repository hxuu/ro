#!/usr/bin/env python3

from dotenv import load_dotenv
import google.generativeai as genai
import PIL.Image
import os
# import httpx
# import base64
import json
from pathlib import Path
# import argparse

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# parser = argparse.ArgumentParser(
#     prog="main.py",
#     description="A versatile tool for reading and solving graph-related problems."
# )
# parser.add_argument(
#     '-i', '--input', required=True, type=str,
#     help="Path to the input file containing the graph data. Supports image or JSON formats."
# )
# parser.add_argument(
#     '-a', '--algorithm', required=True, type=str,
#     help="The algorithm to be applied"
# )
# parser.add_argument(
#     '-v', '--variant', type=str, default=None,
#     help="Specific variant the chosen algorithm (if applicable)."
# )

# Choosing the model
model = genai.GenerativeModel(model_name="gemini-1.5-pro")

# args = parser.parse_args()
# print(args)
# exit()
image_path = "./media/prufer-test.png"
image = PIL.Image.open(image_path)

# image_path = "https://raw.githubusercontent.com/hxuu/ro/refs/heads/main/media/test3.png"
# image = httpx.get(image_path)

# with open('ape.png', 'wb') as f:
#     f.write(image.content)
# exit()
prompt = """
You are an expert in recognizing images of graphs used in graph theory module in college.
You will be given the following tasks:

1. **Task 1: Identify Nodes in the Image**
   - Identify all nodes in the image, represented by circles.
   - Recognize node labels (e.g., A, B, C OR 1, 2, 3 OR S1, S2 etc.) that are attached to the nodes.

2. **Task 2: Detect Edges Between Nodes**
   - Identify edges in the graph, which are lines and/or arrows connecting nodes.
   - Detect if the graph is directed or undirected based on the presence of arrows on the edges.
   - Pay extra attention to the intersections between the lines/arrows

3. **Task 3: Parse Node Labels and Connectivities**
   - For each node, parse its label and the labels of the nodes it is connected to by edges.
   - If the graph is directed, record the direction of the edges.

4. **Task 4: Construct Adjacency List**
   - For each node, create a list of its neighboring nodes (the nodes connected by edges).
   - Include the direction of the edge if the graph is directed.

5. **Task 5: Count the Number of Nodes**
   - Count the total number of unique nodes in the graph based on the parsed labels.

6. **Task 6: Format the Results**
    - Return the result STRICTLY IN THE following format: ```
{
    "num_nodes": number-of-nodes,
    "adjacency_matrix": [
        [ "", "H", "A", "B", "D", "C", "G", "F", "E" ],
        [ "H", 0, 0, 0, 0, 0, 1, 1, 0 ],
        [ "A", 0, 0, 1, 0, 1, 0, 0, 1 ],
        [ "B", 0, 1, 0, 1, 1, 0, 0, 0 ],
        [ "D", 0, 0, 1, 0, 1, 0, 1, 0 ],
        [ "C", 0, 1, 1, 1, 0, 0, 0, 1 ],
        [ "G", 1, 0, 0, 0, 0, 0, 1, 1 ],
        [ "F", 1, 0, 0, 1, 0, 1, 0, 0 ],
        [ "E", 0, 1, 0, 0, 1, 1, 0, 0 ]
    ],
    "adjacency_list": { "H": [
            { "node": "F", "weight": <weight> },
            { "node": "G", "weight": <weight> }...etc
```
"""

# prompt = "answer by yes if you received the image"
# response = model.generate_content([{'mime_type':'image/png', 'data': base64.b64encode(image.content).decode('utf-8')}, prompt])
response = model.generate_content([image, prompt])

# print(response)
import json

def extract_graph_json(response, file_path):
    # access the candidates from the response object
    candidates = response._result.candidates

    if not candidates:
        raise ValueError("no candidates found in the response.")

    # extract the text content from the first candidate
    json_candidate = candidates[0].content.parts[0].text

    # extract the json string between the backticks
    json_start = json_candidate.find('```') + 3
    json_end = json_candidate.rfind('```')
    graph_json_str = json_candidate[json_start:json_end].strip()

    # Debugging: print the extracted JSON string
    print(f"Extracted JSON string: {graph_json_str}")

    if not graph_json_str:
        raise ValueError("Extracted JSON string is empty.")

    with open(file_path, 'w') as f:
        f.write(graph_json_str)


graph_json = extract_graph_json(response,  "./output/prufer-test.json")
