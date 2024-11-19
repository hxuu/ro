import cv2
import pytesseract
import numpy as np

def show(img):
    cv2.imshow("Processed Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def preprocesscrop(img):
    img = cv2.resize(img, None, fx=9, fy=9, interpolation=cv2.INTER_CUBIC)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    show(img)
    return img

def crop_center_region(cropped_region):
    # Convert to grayscale
    gray = cv2.cvtColor(cropped_region, cv2.COLOR_BGR2GRAY)

    # Apply threshold to get binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours (for the text in the cropped region)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the bounding box of the largest contour (assumed to be the text area)
    x_min, y_min, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

    # Optionally add some padding if needed
    padding = 5
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(cropped_region.shape[1], x_min + w + padding)
    y_max = min(cropped_region.shape[0], y_min + h + padding)

    # Crop the image to the bounding box
    cropped_center_region = cropped_region[y_min:y_max, x_min:x_max]

    return cropped_center_region

# Load the image
img = cv2.imread('./media/test3.png')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold the image to get a binary image
_, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

# Detect edges using Canny edge detection
edges = cv2.Canny(binary, threshold1=50, threshold2=150)

# Find contours (edges of the graph)
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Find circles (nodes) - assuming nodes are in circular formation
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                           param1=50, param2=30, minRadius=20, maxRadius=50)

# Process the circles and assign node labels
nodes = {}
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")

    for i, (x, y, r) in enumerate(circles):
        # Crop the region inside the circle
        x1, y1, x2, y2 = x - r, y - r, x + r, y + r
        cropped_region = img[y1+16:y2-16, x1+16:x2-16]
        show(cropped_region)

        # Preprocess and further crop the region to focus on text
        cropped_region = preprocesscrop(cropped_region)

        # Use pytesseract to extract the text (node name) inside the cropped region
        node_name = pytesseract.image_to_string(cropped_region, config='--psm 6').strip()
        print(node_name)

        # Clean up OCR results: remove unwanted characters or symbols
        node_name = ''.join([char for char in node_name if char.isalnum()])

        # If no name was detected, fallback to default naming convention
        if not node_name:
            node_name = f"s{i+1}"

        # Save the node and its position
        nodes[node_name] = (x, y)

        # Draw the circle and the detected name on the image
        cv2.circle(img, (x, y), r, (0, 255, 0), 4)
        cv2.putText(img, node_name, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Use pytesseract to extract weights from the image (numbers over edges)
weights = pytesseract.image_to_string(binary, config='--psm 6')

# Print extracted nodes and weights (optional: debug output)
print("Nodes:", nodes)
print("Weights found:", weights)

# Draw edges and directions (for simplicity, we detect only basic arrows here)
for contour in contours:
    if cv2.isContourConvex(contour):
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 2:  # Simple edge (line)
            cv2.line(img, tuple(approx[0][0]), tuple(approx[1][0]), (0, 0, 255), 2)
        elif len(approx) == 3:  # Arrowhead (directional)
            for i in range(3):
                cv2.line(img, tuple(approx[i][0]), tuple(approx[(i+1)%3][0]), (0, 0, 255), 2)

show(img)

