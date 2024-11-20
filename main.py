#!/usr/bin/env python3

import cv2 as cv
import numpy as np
import pytesseract

OFFSET = 15
MIN_RADIUS = 20

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
    Instead of using distances, connect nodes based on the proximity of line endpoints to node positions.
    """
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

        # Get the coordinates of the closest nodes
        start_node_coords = nodes[closest_node_start]
        end_node_coords = nodes[closest_node_end]

        # Draw the line between the two closest nodes
        cv.line(img, start_node_coords, end_node_coords, (255, 0, 0), 2)  # Red line

        # Optionally, print the node names and their coordinates for debugging
        print(f"Connecting '{closest_node_start}' with '{closest_node_end}'")

    return img


def main():
    FILEPATH = './media/test3.png'

    original_img = cv.imread(FILEPATH)

    gray = cv.cvtColor(original_img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (9, 9), 2)
    processed_img = blur

    # Detect and label the circles
    nodes, labeled_img = extract_and_label_circles(original_img, processed_img)

    for node_name, (x, y) in nodes.items():
        print(f"Node '{node_name}' found at coordinates: ({x}, {y})")

    # Detect lines using Probabilistic Hough Line Transform
    lines, result_img = detect_lines(original_img)

    # img_show(result_img)
    # Connect nodes using detected lines
    result_img = connect_nodes_with_lines(nodes, lines, result_img)

    # Display the result image with connected nodes
    img_show(result_img)

if __name__ == '__main__':
    main()

