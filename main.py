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
image_path = "./output/ape.png"
image = PIL.Image.open(image_path)

# image_path = "https://raw.githubusercontent.com/hxuu/ro/refs/heads/main/media/test3.png"
# image = httpx.get(image_path)

# with open('ape.png', 'wb') as f:
#     f.write(image.content)
# exit()
prompt = """
Analyze the given graph image carefully and extract the following information with attention to detail:

1. **Vertices**:
   - List all the unique vertex names (nodes) displayed in the image.

2. **Edges**:
   - Specify if the graph is **directed** or **undirected**.
   - For each edge, consider the direction
   - Extract **weights** (if any) that are displayed next to or on top of the edges.
   - Pay close attention to intersections or overlapping edges to avoid misattributions.
   - Pay close attention to small arrows at the end of arcs if the graph is directed.
   For example a graph can have multiple incoming arrows but one outgoing arrow.

3. **Adjacency List**:
   - Build an adjacency list representing the graph, showing neighbors for each vertex.
   - Include weights if applicable.

4. **Adjacency Matrix**:
   - Construct the adjacency matrix for the graph.
   - If weights exist, the matrix should display the weights; otherwise, use binary values (0 for no edge, 1 for an edge).

5. **Graph Type**:
   - Identify the graph type:
     - Directed or undirected.
     - Weighted or unweighted.

6. **Edge Directions**:
   - Highlight all directed edges with arrows.
   - Double-check all intersections or closely overlapping edges to determine correct edge directions.

**Notes**:
- Prioritize accuracy when identifying edges and directions, especially where multiple lines intersect or overlap.
- Ensure all vertices and edges are accounted for, and no connections are missed.
- If any ambiguity exists, describe it clearly.

Return the result STRICTLY IN THE following format: ```
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

print(response)

