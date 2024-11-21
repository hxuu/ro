#!/usr/bin/env python3

import re
import pytesseract
import json
import os
import cv2 as cv
import numpy as np

OFFSET = 15
MIN_RADIUS = 20
FILEPATH = './media/test2.png'
OUTPUT_DIR = './output'

# Ensure output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def img_show(img):
    """Display an image."""
    cv.imshow("Image", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def img_preprocess(img):
    """Preprocess the input image for arrow detection."""
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 2)
    canny = cv.Canny(blur, 50, 50)

    kernel = np.ones((2, 2))

    dilate = cv.dilate(canny, kernel, iterations=2)
    erode = cv.erode(dilate, kernel, iterations=1)

    return erode

def crop_preprocess(cropped_img):
    scale = cv.resize(cropped_img, None, fx=9, fy=9, interpolation=cv.INTER_CUBIC)
    blur = cv.bilateralFilter(scale, 9, 75, 75)
    return blur

def extract_and_label_circles(img, processed_img):
    """
    Detect circles, extract their content, use pytesseract to read node names,
    and draw the circles with labels on the image.
    """
    nodes = {}

    # Detect circles
    circles = cv.HoughCircles(
        processed_img,
        cv.HOUGH_GRADIENT,
        dp=0.9,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=MIN_RADIUS,
        maxRadius=100
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        for i, (x, y, r) in enumerate(circles):
            # Define the bounding box for the circle and crop the region
            x1, y1, x2, y2 = x - r, y - r, x + r, y + r
            cropped_region = img[y1+OFFSET:y2-OFFSET, x1+OFFSET:x2-OFFSET]
            cropped_preprocessed = crop_preprocess(cropped_region)

            node_name = pytesseract.image_to_string(cropped_preprocessed, config='--psm 9').strip()
            node_name = ''.join([char for char in node_name if char.isalnum()])

            # Handle some special cases for node names
            if node_name.lower() == 'ST'.lower():
                node_name = 'S7'
            if node_name.lower() == 'S92'.lower():
                node_name = 'S2'
            if node_name.lower() == ''.lower():
                node_name = 'G'

            nodes[node_name] = (x, y)

            # Draw the circle and label on the image
            cv.circle(img, (x, y), r, (0, 255, 0), 4)
            cv.putText(img, node_name, (x - 10, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return nodes, img

def detect_lines(img):
    """Detect lines using Probabilistic Hough Line Transform."""
    # Convert the image to grayscale and apply edge detection (Canny)
    edges = img_preprocess(img)

    img_show(edges)
    # Detect lines using the Probabilistic Hough Line Transform
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=2)

    if lines is not None:
        i = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw the line in red color
            i += 1

    return lines, img

def connect_nodes_with_lines(nodes, lines, img):
    """
    Connect nodes using the lines detected by HoughLinesP.
    Returns the adjacency matrix (with node names), adjacency list, and updated image.
    """
    connected = set()  # Set to store unique connections
    adjacency_list = {node: [] for node in nodes}  # Initialize adjacency list
    node_names = list(nodes.keys())  # List of node names for matrix indexing
    num_nodes = len(node_names)
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)  # Initialize adjacency matrix

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Find the closest node to the line's start point (x1, y1)
        closest_node_start = None
        min_dist_start = float('inf')
        for node_name, (nx, ny) in nodes.items():
            dist = np.linalg.norm(np.array([x1, y1]) - np.array([nx, ny]))
            if dist < min_dist_start:
                min_dist_start = dist
                closest_node_start = node_name

        # Find the closest node to the line's end point (x2, y2)
        closest_node_end = None
        min_dist_end = float('inf')
        for node_name, (nx, ny) in nodes.items():
            dist = np.linalg.norm(np.array([x2, y2]) - np.array([nx, ny]))
            if dist < min_dist_end:
                min_dist_end = dist
                closest_node_end = node_name

        # Avoid self-loop (same node connected to itself)
        if closest_node_start == closest_node_end:
            continue

        # Check if the connection is already added to prevent duplicates
        connection = tuple(sorted([closest_node_start, closest_node_end]))
        if connection in connected:
            continue

        # Draw the line between the two closest nodes
        start_node_coords = nodes[closest_node_start]
        end_node_coords = nodes[closest_node_end]
        cv.line(img, start_node_coords, end_node_coords, (255, 0, 0), 2)  # Red line

        # Update adjacency list
        adjacency_list[closest_node_start].append({"node": closest_node_end, "weight": 1})  # Default weight
        adjacency_list[closest_node_end].append({"node": closest_node_start, "weight": 1})  # Default weight

        # Update adjacency matrix
        start_idx = node_names.index(closest_node_start)
        end_idx = node_names.index(closest_node_end)
        adjacency_matrix[start_idx][end_idx] = 1
        adjacency_matrix[end_idx][start_idx] = 1

        # Add the connection to the set
        connected.add(connection)

    # Add node names to adjacency matrix
    matrix_with_names = [[""] + node_names]  # Top row with column names
    for i, node_name in enumerate(node_names):
        matrix_with_names.append([node_name] + adjacency_matrix[i].tolist())  # Add row with row name

    return matrix_with_names, adjacency_list, img


def pretty_json(data):
    """
    With the power of regex, remove the annoying \n inside an array in `le Jeson`
    Our input (json output) looks like this:
        {
        |    "num_nodes": 8,
        |    "adjacency_matrix": [
        |        [
        |            "",
        |            "H",
        |        ],
        |        [
        |            "E",
        |            0,
        |       ]
        |    ],
        |    "adjacency_list": {
        |    |   "H": [
        |    |       {
        |    |           "node": "F",
        |    |           "weight": 1
        |    |       },
        |    |       {
        |    |           "node": "G",
        |    |           "weight": 1
        |    |       }
        |    |   ],
        |    }
        }
    We'll turn this boi into something more readable
    """

    # define the necessary matches
    SBRACKET = r'\[\n\s+"'
    CBRACKET = r'\{\n\s+'

    C_INSIDE = r'",\n\s+'
    C_INSIDE_LF = r'(")\n\s+\]'

    D_INSIDEARR = r'(\d,)\n\s+'
    D_INSIDEARR_LF = r'(\d)\n\s+\]'
    D_INSIDEOBJ_LF = r'(\d)\n\s+\}'

    # replace all occurrences
    data = re.sub(SBRACKET, '[ "', data)
    data = re.sub(C_INSIDE, '", ', data)
    data = re.sub(D_INSIDEARR, r'\1 ', data)
    data = re.sub(D_INSIDEARR_LF, r'\1 ]', data)
    data = re.sub(C_INSIDE_LF, r'\1 ]', data)
    data = re.sub(CBRACKET, '{ ', data)
    data = re.sub(D_INSIDEOBJ_LF, r'\1 }', data)

    return data


def save_graph_data(matrix_with_names, adjacency_list, output_dir='./output'):
    """
    Save adjacency matrix, adjacency list, and number of nodes into one JSON file.
    """
    # Extract the node names from the first row of the matrix_with_names
    node_names = matrix_with_names[0][1:]
    num_nodes = len(node_names)

    # Combine all data
    graph_data = {
        "num_nodes": num_nodes,
        "adjacency_matrix": matrix_with_names,
        "adjacency_list": adjacency_list
    }

    pretty_graph_data = pretty_json(json.dumps(graph_data, indent=4))

    # Save to JSON file
    filename = f"{FILEPATH.split('/')[-1].split('.')[0]}.json"
    print(filename)
    output_file_path = os.path.join(output_dir, filename)
    with open(output_file_path, "w") as f:
        f.write(pretty_graph_data)

    print(f"Graph data saved to {output_file_path}")


def main():
    original_img = cv.imread(FILEPATH)

    gray = cv.cvtColor(original_img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (9, 9), 2)
    processed_img = blur

    # Detect and label the circles
    nodes, labeled_img = extract_and_label_circles(original_img, processed_img)

    for node_name, (x, y) in nodes.items():
        print(f"Node '{node_name}' found at coordinates: ({x}, {y})")

    lines, result_img = detect_lines(original_img)

    matrix_with_names, adjacency_list, result_img_with_connections = connect_nodes_with_lines(nodes, lines, original_img)

    img_show(result_img_with_connections)

    save_graph_data(matrix_with_names, adjacency_list)


if __name__ == '__main__':
    main()

