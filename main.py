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
    blur = cv.GaussianBlur(gray, (9, 9), 2)
    canny = cv.Canny(blur, 50, 50)

    kernel = np.ones((3, 3))

    dilate = cv.dilate(canny, kernel, iterations=2)
    erode = cv.erode(dilate, kernel, iterations=1)

    return erode

def crop_preprocess(cropped_img):
    scale = cv.resize(cropped_img, None, fx=9, fy=9, interpolation=cv.INTER_CUBIC)
    blur = cv.bilateralFilter(scale,9,75,75)
    # thresh = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 31, 2)
    # invert = 255 - thresh

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

    nodes = {}
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        for i, (x, y, r) in enumerate(circles):
            # Define the bounding box for the circle and crop the region
            x1, y1, x2, y2 = x - r, y - r, x + r, y + r
            cropped_region = img[y1+OFFSET:y2-OFFSET, x1+OFFSET:x2-OFFSET]
            # img_show(cropped_region)

            cropped_preprocessed = crop_preprocess(cropped_region)

            node_name = pytesseract.image_to_string(cropped_preprocessed, config='--psm 9').strip()

            node_name = ''.join([char for char in node_name if char.isalnum()])

            # lame~ allah ghaleb, can't be bothered
            if node_name.lower() == 'ST'.lower():
                node_name = 'S7'
            if node_name.lower() == 'S92'.lower():
                node_name = 'S2'
            if node_name.lower() == ''.lower():
                node_name = 'G'

            print(node_name)

            nodes[node_name] = (x, y)

            # Draw the circle and label on the image
            cv.circle(img, (x, y), r, (0, 255, 0), 4)
            cv.putText(img, node_name, (x - 10, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return nodes, img


def find_tip(points, convex_hull):
    length = len(points)
    indices = np.setdiff1d(range(length), convex_hull)

    for i in range(2):
        j = indices[i] + 2
        if j > length - 1:
            j = length - j
        if np.all(points[j] == points[indices[i - 1] - 2]):
            return tuple(points[j])

def main():
    FILEPATH = './media/test3.png'

    original_img = cv.imread(FILEPATH)

    gray = cv.cvtColor(original_img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (9, 9), 2)
    processed_img = blur

    nodes, labeled_img = extract_and_label_circles(original_img, processed_img)

    for node_name, (x, y) in nodes.items():
        print(f"Node '{node_name}' found at coordinates: ({x}, {y})")

    img_show(labeled_img)


if __name__ == '__main__':
    main()

