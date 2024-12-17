#!/usr/bin/env python3

from dotenv import load_dotenv
import google.generativeai as genai
import PIL.Image
import os
import httpx
import base64
import argparse

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

parser = argparse.ArgumentParser(
    prog="main.py",
    description="A versatile tool for reading and solving graph-related problems."
)
parser.add_argument(
	'-i', '--input', required=True, type=str,
	help="Path to the input file containing the graph data. Supports image or JSON formats."
)
parser.add_argument(
	'-a', '--algorithm', required=True, type=str,
	help="The algorithm to be applied"
)
parser.add_argument(
	'-v', '--variant', type=str, default=None,
	help="Specific variant the chosen algorithm (if applicable)."
)

# Choosing the model
model = genai.GenerativeModel(model_name="gemini-1.5-flash-8b")

args = parser.parse_args()
print(args)
exit()
image_path = "./media/test3.png"
image = PIL.Image.open(image_path)

# image_path = "https://raw.githubusercontent.com/hxuu/ro/refs/heads/main/media/test3.png"
# image = httpx.get(image_path)

# with open('ape.png', 'wb') as f:
#     f.write(image.content)
# exit()
prompt = """
Extract an adjancency list out of this image of a graph. Extract the vertices names
out of the nodes drawen, pay attention to the direction of the edges, the graph could
be either directed or non-directed, and extract the weights that are next to (or on top)
of the edges between the vertices if they exists.

Return the result following this form: ```
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
            { "node": "G", "weight": <weight> }...
```
"""

# prompt = "answer by yes if you received the image"
# response = model.generate_content([{'mime_type':'image/png', 'data': base64.b64encode(image.content).decode('utf-8')}, prompt])
response = model.generate_content([image, prompt])

print(response)


