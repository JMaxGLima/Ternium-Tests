import cv2
from tqdm import tqdm

CANNY_WINDOW_TITLE = 'Edge'

def sketch_transform(image):
    # Convert to grayscale
    image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Canny edge detector
    image_canny = cv2.Canny(image_grayscale, 8000, 10000, apertureSize=7)
    # Invert the Canny edges to create a sketch effect
    _, mask = cv2.threshold(image_canny, 100, 255, cv2.THRESH_BINARY_INV)
    return mask

# Open video file
cam_capture = cv2.VideoCapture("/home/max/Videos/testCam/WoodTest3.avi")

if not cam_capture.isOpened():
    print("Error: Could not open video.")
    exit()

# Define the region of interest (ROI) coordinates
upper_left = (100, 100)
bottom_right = (700, 700)

total_frames = int(cam_capture.get(cv2.CAP_PROP_FRAME_COUNT))
progress_bar = tqdm(total=total_frames, unit='frame', desc='Processing')

while True:
    ret, image_frame = cam_capture.read()
    
    if not ret:
        print("Error: Could not read frame.")
        break

    # Draw a rectangle marker
    cv2.rectangle(image_frame, upper_left, bottom_right, (0, 255, 0), 5)
    
    # Extract the region of interest (ROI)
    rect_img = image_frame[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]
    
    # Apply the sketch transformation
    sketcher_rect = sketch_transform(rect_img)
    
    # Convert sketch to RGB
    sketcher_rect_rgb = cv2.cvtColor(sketcher_rect, cv2.COLOR_GRAY2RGB)
    
    # Replace the ROI in the original frame with the sketched image
    image_frame[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]] = sketcher_rect_rgb
    
    # Display the video with sketch effect
    cv2.imshow("Video", image_frame)
    progress_bar.update(1)

    key = cv2.waitKey(30) & 0xFF
    if key == 27:  
        break
    elif key == ord('p'): 
        cv2.waitKey(-1)  

# Release the video capture and close all windows
progress_bar.close()
cam_capture.release()
cv2.destroyAllWindows()
