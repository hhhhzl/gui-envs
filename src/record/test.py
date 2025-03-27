import cv2


# Function to merge the resized ROI back into the original frame
def merge_roi(frame, roi, x, y):
    frame[y:y + 200, x:x + 200, :] = roi
    return frame


# Capture video from the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    # If the frame was not read successfully, break the loop
    if not ret:
        break

    # Crop a region of interest (ROI) from the frame
    roi = frame[150:400, 200:450]

    # Resize the ROI to a specific size (e.g., 200x200)
    roi_resized = cv2.resize(roi, (200, 200))

    # Merge the resized ROI back into the frame
    frame = merge_roi(frame, roi_resized, 0, 0)

    # Display the merged frame
    cv2.imshow("Merged Video", frame)

    # Check for the 'q' key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the display window
cap.release()
cv2.destroyAllWindows()
