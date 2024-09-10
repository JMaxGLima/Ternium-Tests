import cv2
import numpy as np

# Define a function to process each frame
def process_frame(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Hough Transform to detect lines (cracks or linear defects)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=1000, maxLineGap=1000)

    # Draw detected lines
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Keypoint Detection using ORB
    orb = cv2.ORB_create(nfeatures=10000000)
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    # Draw detected keypoints
    keypoint_image = cv2.drawKeypoints(frame, keypoints, None, color=(0, 0, 255))

    # Segmentation using Thresholding
    _, thresholded = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV)

    # Apply Morphological Operations to clean up the segmented image
    kernel = np.ones((5, 5), np.uint8)
    morph = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)

    # Find contours on segmented image
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the image
    segmentation_result = cv2.drawContours(frame.copy(), contours, -1, (255, 0, 0), 2)

    return frame, keypoint_image, segmentation_result

# Open the video file or capture from a webcam (use 0 for webcam)
cap = cv2.VideoCapture("/home/max/Documents/Ternium-Tests/Database/testCam/WoodTest0_100cm_123_.avi")  # Replace 'wood_video.mp4' with 0 for webcam

# Check if the video capture opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    # Read frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break  # Exit the loop if no frame is captured

    # Process the current frame
    frame_result, keypoint_image, segmentation_result = process_frame(frame)

    # Display results
    cv2.imshow('Original Frame with Hough Transform', frame_result)
    cv2.imshow('Keypoints Detected with ORB', keypoint_image)
    cv2.imshow('Segmentation Result', segmentation_result)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
