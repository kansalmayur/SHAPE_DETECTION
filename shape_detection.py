import cv2
import numpy as np

def detect_and_measure_shapes(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image.")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(thresh, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue

        perimeter = cv2.arcLength(contour, True)
        
        epsilon = 0.04 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)

        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(approx)

        shape_type = "Polygon"
        if len(approx) == 3:
            shape_type = "Triangle"
        elif len(approx) == 4:
            aspect_ratio = float(w) / h
            if 0.95 <= aspect_ratio <= 1.05:
                shape_type = "Square"
            else:
                shape_type = "Rectangle"
        elif len(approx) == 6:
            shape_type = "Hexagon"

        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        cv2.drawContours(img, [approx], 0, (0, 255, 0), 3)

        label = f"{shape_type} (A:{int(area)} P:{int(perimeter)})"
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("Shape Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_and_measure_shapes('hexagon.jpg')
