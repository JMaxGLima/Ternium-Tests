"""
////////////////////////////////////////////////////////////////////////////////////

Objective:
This script is designed to process video files and apply a sketch effect to a selected region of interest (ROI) within each frame. 
It utilizes computer vision techniques such as Canny edge detection and Hough Line Transform  on the selected ROI. 

The program allows the user to manually select a ROI on the first frame of the video. After the ROI is selected, 
the script processes each frame of the video by applying the Canny detection and the Hough Transform to the chosen region. 
The result is displayed in real-time with a progress bar indicating the processing status.

Prerequisites:
- Make sure that the OpenCV (`cv2`), NumPy (`np`), and `tqdm` libraries are installed in your Python environment.
- The script should be run in an environment that supports OpenCV's GUI functions, such as a local machine with a graphical interface.
- Provide the correct path to the input video file in the `cv2.VideoCapture` function call.

////////////////////////////////////////////////////////////////////////////////////
"""

import cv2  # Import the OpenCV library for computer vision tasks
import numpy as np  # Import NumPy for numerical operations (though not used in this snippet)
from tqdm import tqdm  # Import tqdm for displaying a progress bar

# Constants for window titles
ORIGINAL_WINDOW_TITLE = 'Original'
CANNY_WINDOW_TITLE = 'Edge'

def initialize_camera(cap):
    """
    Skips the first 60 frames of the video capture to allow the camera to adjust
    and then retrieves the first usable frame.
    
    Args:
        cap (cv2.VideoCapture): The video capture object.
    
    Returns:
        frame (numpy.ndarray): The first usable frame after skipping initial frames.
    
    Raises:
        Exception: If a frame cannot be retrieved from the video capture.
    """
    for i in range(60):  # Skip the initial frames
        cap.read()  # Read and discard frames
    ret, frame = cap.read()  # Read the next frame
    if not ret:
        raise Exception("Failed to retrieve a frame from the video.")
    return frame

def nothing(*arg):
    """
    A placeholder function for trackbar callbacks. It does nothing and is used
    to satisfy the OpenCV requirement for a callback function.
    
    Args:
        *arg: Variable length argument list (not used in this function).
    """
    pass

def sketch_transform(image, low_threshold, high_threshold):
    """
    Applies a sketch effect to the input image using Canny edge detection and contour drawing.
    
    Args:
        image (numpy.ndarray): The input image on which the sketch effect will be applied.
        low_threshold (int): The lower threshold for the Canny edge detector.
        high_threshold (int): The upper threshold for the Canny edge detector.
    
    Returns:
        image_copy (numpy.ndarray): The image with the sketch effect applied.
    """
    # Convert the image to grayscale
    image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur to the grayscale image to reduce noise
    image_blur = cv2.GaussianBlur(image_grayscale, (5, 5), 0)  
    
    # Apply Canny edge detector to the blurred grayscale image
    image_canny = cv2.Canny(image_blur, low_threshold, high_threshold, apertureSize=7)  
    
    # _, mask = image_canny_inverted = cv2.threshold(image_canny, 100, 255, cv2.THRESH_BINARY_INV)
    
    # Define a kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    
    # Close gaps in edges and remove small noise using morphological operations
    mask = cv2.morphologyEx(image_canny, cv2.MORPH_CLOSE, kernel)  
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  

    # Find contours in the binary image (sketch mask)
    # contours, hierarchy = cv2.findContours(image=mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)                                 
    
    # Create a copy of the original image to draw contours on
    image_copy = image.copy()
    
    # Detect lines using Hough Line Transform
    lines = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=100, minLineLength=1000, maxLineGap=1000)
    
    # Draw detected lines on the copied image
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image_copy, (x1, y1), (x2, y2), (0, 0, 255), 4)  # Draw lines in red

    return image_copy

def select_roi(event, x, y, flags, param):
    """
    Callback function to handle mouse events for selecting a Region of Interest (ROI).
    
    Args:
        event (int): The event type (e.g., mouse click).
        x (int): The x-coordinate of the mouse event.
        y (int): The y-coordinate of the mouse event.
        flags (int): The flags associated with the event.
    """
    global upper_left, bottom_right, selecting, roi_selected

    if event == cv2.EVENT_LBUTTONDOWN:
        # Start selecting the ROI
        upper_left = (x, y)  # Top-left corner of the ROI
        selecting = True
        roi_selected = False
    elif event == cv2.EVENT_MOUSEMOVE:
        if selecting:
            # Update the bottom-right corner of the ROI
            bottom_right = (x, y)
            # Draw the rectangle on the frame to show the selected ROI
            temp_frame = first_frame.copy()  # Copy the first frame to draw the ROI rectangle
            cv2.rectangle(temp_frame, upper_left, bottom_right, (0, 255, 0), 5)  # Draw rectangle in green
            cv2.imshow("Select ROI", temp_frame)  # Display the frame with the ROI rectangle
    elif event == cv2.EVENT_LBUTTONUP:
        # Finish selecting the ROI
        bottom_right = (x, y)
        selecting = False
        roi_selected = True

# Initialize global variables for ROI selection
upper_left = (0, 0)
bottom_right = (0, 0)
selecting = False
roi_selected = False

# Open video file
cam_capture = cv2.VideoCapture("/home/max/Documents/Ternium-Tests/Database/testCam/WoodTest2_Vert_0d.avi")

if not cam_capture.isOpened():
    print("Error: Could not open video.")
    exit()

# Retrieve the first frame from the video to initialize ROI selection
first_frame = initialize_camera(cam_capture)

# Create a window for ROI selection and set the mouse callback function
cv2.namedWindow("Select ROI", cv2.WINDOW_KEEPRATIO)
cv2.setMouseCallback("Select ROI", select_roi)

# Display the first frame to allow the user to select ROI
while not roi_selected:
    cv2.imshow("Select ROI", first_frame)
    key = cv2.waitKey(10)
    # Press Enter to break the loop
    if key == 13:
        break
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to cancel ROI selection
        print("ROI selection cancelled.")
        cam_capture.release()
        cv2.destroyAllWindows()
        exit()

# If ROI was not selected, use the entire frame
if not roi_selected:
    print('ROI Not Selected. Using Full Frame')
    upper_left = (0, 0)
    bottom_right = (first_frame.shape[1] - 1, first_frame.shape[0] - 1)

# Create a progress bar to show the processing status
total_frames = int(cam_capture.get(cv2.CAP_PROP_FRAME_COUNT))  # Get the total number of frames in the video
progress_bar = tqdm(total=total_frames, unit='frame', desc='Processing')

# Create a window for trackbars to adjust Canny edge detection thresholds
cv2.namedWindow(CANNY_WINDOW_TITLE, cv2.WINDOW_KEEPRATIO)
cv2.createTrackbar('thrs1', CANNY_WINDOW_TITLE, 15000, 30000, nothing)  # Trackbar for lower threshold
cv2.createTrackbar('thrs2', CANNY_WINDOW_TITLE, 20000, 30000, nothing)  # Trackbar for upper threshold

while True:
    ret, image_frame = cam_capture.read()
    
    if not ret:
        print("Error: Could not read frame.")
        break

    # Draw a rectangle marker on the video frame to indicate the ROI
    if upper_left[0] < bottom_right[0] and upper_left[1] < bottom_right[1]:
        cv2.rectangle(image_frame, upper_left, bottom_right, (0, 255, 0), 5)
        
        # Extract the region of interest (ROI) from the frame
        rect_img = image_frame[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]
        
        # Get the current positions of the trackbars for threshold values
        low_threshold = cv2.getTrackbarPos('thrs1', CANNY_WINDOW_TITLE)
        high_threshold = cv2.getTrackbarPos('thrs2', CANNY_WINDOW_TITLE)
        
        # Apply the sketch transformation to the extracted ROI
        sketcher_rect = sketch_transform(rect_img, low_threshold, high_threshold)
        
        # Convert the sketch to RGB format for proper display
        sketcher_rect_rgb = cv2.cvtColor(sketcher_rect, cv2.COLOR_BGR2RGB)
        
        # Replace the ROI in the original frame with the sketched image
        image_frame[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]] = sketcher_rect_rgb

    # Display the video with the sketch effect applied to the ROI
    cv2.imshow(CANNY_WINDOW_TITLE, image_frame)
    progress_bar.update(1)  # Update the progress bar with each processed frame

    key = cv2.waitKey(30) & 0xFF
    if key == 27:  # ESC key to exit
        break
    elif key == ord('p'):  # 'p' key to pause
        cv2.waitKey(-1)  # Wait indefinitely until a key is pressed

# Release the video capture and close all windows
progress_bar.close()  # Close the progress bar
cam_capture.release
