import cv2
import numpy as np
from tqdm import tqdm

ORIGINAL_WINDOW_TITLE = 'Original'
CANNY_WINDOW_TITLE = 'Edge'

def initialize_camera(cap):
    """ Skips the first 60 frames and returns the first usable frame. """
    cap.set(cv2.CAP_PROP_POS_FRAMES, 60)
    ret, frame = cap.read()
    if not ret:
        raise Exception("Failed to retrieve a frame from the video.")
    return frame

def nothing(*arg):
    pass

import cv2
import numpy as np

def detect_deformation(frame, lines):
    deformation_detected = False

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

            if 2 < abs(angle) < 10:
                deformation_detected = True
                # Draw the deformed line 
                cv2.line(frame, (x1, y1), (x2, x2), (0, 255, 255), 3)
                print(f"Deformation Detected: {angle:.2f} degrees")

    # If deformation detected, pause the video
    if deformation_detected:
        print("Deformation detected! Pausing. Press 'c' to continue or 'q' to quit.")
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('c'):  
                break
            elif key == ord('q'): 
                cam_capture.release()
                cv2.destroyAllWindows()
                exit()
    
    return frame

def sketch_transform(image, low_threshold, high_threshold):
    """Applies sketch effect using Canny edge detection and Shi-Tomasi corner detection."""
    # Convert to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, low_threshold, high_threshold, apertureSize=7)

    # Apply morphological transformations to clean the edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)

    # Detect lines using Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=100)

    # Create a copy of the original image for line drawing
    image_copy = image.copy()

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Draw lines in green
            cv2.line(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 4)

    # Apply Shi-Tomasi corner detection
    corners = cv2.goodFeaturesToTrack(blurred, maxCorners=50, qualityLevel=0.1, minDistance=10)
    corners = np.int32(corners)

    # Detect deformation and update the image with deformed lines
    image_copy = detect_deformation(image_copy, lines)

    # Draw red circles at detected corners
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(image_copy, (x, y), 5, (0, 0, 255), -1)  # Red dot at each corner

    return image_copy


def select_roi(event, x, y, flags, param):
    """ Callback function to handle mouse events for selecting a ROI. """
    global upper_left, bottom_right, selecting, roi_selected

    if event == cv2.EVENT_LBUTTONDOWN:
        upper_left = (x, y)
        selecting = True
        roi_selected = False
    elif event == cv2.EVENT_MOUSEMOVE and selecting:
        bottom_right = (x, y)
        temp_frame = first_frame.copy()
        cv2.rectangle(temp_frame, upper_left, bottom_right, (255, 0, 0), 5)
        cv2.imshow("Select ROI", temp_frame)
    elif event == cv2.EVENT_LBUTTONUP:
        bottom_right = (x, y)
        selecting = False
        roi_selected = True

# Initialize global variables for ROI selection
upper_left = (0, 0)
bottom_right = (0, 0)
selecting = False
roi_selected = False

# Open video file
video_path = "/home/max/Documents/Ternium-Tests/Database/testCam/WoodTestZED0_200cm_123_.avi"
cam_capture = cv2.VideoCapture(video_path)

if not cam_capture.isOpened():
    print("Error: Could not open video.")
    exit()

# Retrieve the first frame for ROI selection
first_frame = initialize_camera(cam_capture)

# Window for ROI selection
cv2.namedWindow("Select ROI", cv2.WINDOW_KEEPRATIO)
cv2.setMouseCallback("Select ROI", select_roi)

while not roi_selected:
    cv2.imshow("Select ROI", first_frame)
    if cv2.waitKey(10) == 13:  # Enter key to confirm ROI
        break
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to cancel ROI selection
        print("ROI selection cancelled.")
        cam_capture.release()
        cv2.destroyAllWindows()
        exit()

if not roi_selected:
    print('ROI Not Selected. Using Full Frame')
    upper_left = (0, 0)
    bottom_right = (first_frame.shape[1] - 1, first_frame.shape[0] - 1)

# Progress bar
total_frames = int(cam_capture.get(cv2.CAP_PROP_FRAME_COUNT))
progress_bar = tqdm(total=total_frames, unit='frame', desc='Processing')

# Canny edge detection thresholds
cv2.namedWindow(CANNY_WINDOW_TITLE, cv2.WINDOW_KEEPRATIO)
cv2.createTrackbar('thrs1', CANNY_WINDOW_TITLE, 15000, 30000, nothing)
cv2.createTrackbar('thrs2', CANNY_WINDOW_TITLE, 20000, 30000, nothing)

while True:

    if not cam_capture.isOpened():
        break

    ret, image_frame = cam_capture.read()
    
    if not ret:
        cam_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        progress_bar.reset()
        continue

    # Draw the ROI rectangle
    if upper_left[0] < bottom_right[0] and upper_left[1] < bottom_right[1]:
        cv2.rectangle(image_frame, upper_left, bottom_right, (255, 0, 0), 5)
        rect_img = image_frame[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]
        
        # Get Canny thresholds from the trackbars
        low_threshold = cv2.getTrackbarPos('thrs1', CANNY_WINDOW_TITLE)
        high_threshold = cv2.getTrackbarPos('thrs2', CANNY_WINDOW_TITLE)
        
        # Apply sketch transformation
        sketcher_rect = sketch_transform(rect_img, low_threshold, high_threshold)
        
        # Replace ROI in the original frame with the sketched result
        image_frame[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]] = sketcher_rect

    # Display the processed frame
    cv2.imshow(CANNY_WINDOW_TITLE, image_frame)
    progress_bar.update(1)  # Update progress bar

    key = cv2.waitKey(30) & 0xFF
    if key == 27 or key == ord('q') :  # ESC or Q key to exit
        break
    elif key == ord('p'):  # 'p' key to pause
        cv2.waitKey(-1)  # Wait indefinitely for any key press

progress_bar.close()
cam_capture.release()
cv2.destroyAllWindows()
