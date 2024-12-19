#!/usr/bin/env python3

from dotenv import load_dotenv
from utils.arg_parser import get_argument_parser
from utils.extract import extract_graph_json
from algorithms import *

import google.generativeai as genai
import PIL.Image
import os
from pprint import pprint
import time

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Choosing the model
model = genai.GenerativeModel(model_name="gemini-1.5-pro")

# Suppress logging warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

parser = get_argument_parser()
args = parser.parse_args()

# Load image and show progress
image_path = args.input
print("[-] Loading image from", image_path, "...\n |")
image = PIL.Image.open(image_path)
print("[+] Image loaded successfully.")

# Read the prompt from file
with open("./prompt.txt", 'r') as file:
    prompt = file.read()
print("[+] Prompt loaded successfully.")

# Generate response from the model with a progress bar
print("[-] Generating content using the model...\n |")
response = model.generate_content([image, prompt])
print("[+] Content generated.")

# Extract graph JSON with a progress bar
json_path = "./output/" + os.path.basename(image_path).split(".")[0] + ".json"
print(f"[-] Extracting graph to {json_path}...\n |")
extract_graph_json(response, json_path)
print("[+] Graph extracted successfully.")

# Select and execute the chosen algorithm
algorithm = args.algorithm
print(f"[-] Selected algorithm: {algorithm}\n |")

match algorithm:
    case "bfs":
        start_vertex = "change me..."
        print(f"[-] Running BFS starting from {start_vertex}...")
        bfs(json_path, start_vertex)
        print("[+] BFS execution complete.")

    case "dfs":
        start_vertex = "change me..."
        print(f"[-] Running DFS starting from {start_vertex}...")
        dfs(json_path, start_vertex)
        print("[+] DFS execution complete.")

    case "prufer_encoding":
        print("[-] Running Prufer encoding...")
        prufer_encoding(json_path)
        print("[+] Prufer encoding complete.")

    case "prufer_decoding":
        print("[-] Running Prufer decoding...")
        prufer_decoding(json_path)
        print("[+] Prufer decoding complete.")

    case "kruskal":
        print("[-] Running Kruskal's algorithm...")
        kruskal(json_path)
        print("[+] Kruskal's algorithm complete.")

    case _:
        available_algorithms = ['bfs', 'dfs', 'prufer_encoding', 'prufer_decoding', 'kruskal']
        print("Make sure your algorithm is in the list of available algorithms!")
        pprint(available_algorithms)
