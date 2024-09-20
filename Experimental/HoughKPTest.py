import cv2
import numpy as np

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=1000, maxLineGap=1000)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    orb = cv2.ORB_create(nfeatures=10000000)
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    keypoint_image = cv2.drawKeypoints(frame, keypoints, None, color=(0, 0, 255))

    _, thresholded = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((5, 5), np.uint8)
    morph = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    segmentation_result = cv2.drawContours(frame.copy(), contours, -1, (255, 0, 0), 2)

    return frame, keypoint_image, segmentation_result

cap = cv2.VideoCapture("/home/max/Documents/Ternium-Tests/Database/testCam/WoodTest2_Vert_0d.avi")  

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break 

    frame_result, keypoint_image, segmentation_result = process_frame(frame)

    cv2.imshow('Original Frame with Hough Transform', frame_result)
    cv2.imshow('Keypoints Detected with ORB', keypoint_image)
    cv2.imshow('Segmentation Result', segmentation_result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
