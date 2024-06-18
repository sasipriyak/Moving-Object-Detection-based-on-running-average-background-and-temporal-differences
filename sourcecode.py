import cv2
import numpy as np

def initialize_background(frame, alpha=0.5):
    # Use the first frame as the initial background
    return frame.astype(float)

def update_background(bg_image, current_frame, alpha=0.02):
    # Update the background model with a running average
    return (alpha * current_frame + (1 - alpha) * bg_image).astype(float)

def apply_background_subtraction(bg_image, current_frame, threshold=25):
    # Subtract background and get a binary image
    diff = cv2.absdiff(bg_image.astype(np.uint8), current_frame)
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    return thresh

def main():
    cap = cv2.VideoCapture(0)  # Change to cv2.VideoCapture('filename.mp4') for a file
    ret, frame = cap.read()
    if not ret:
        print("Failed to read video")
        return

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bg_image = initialize_background(gray_frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Background subtraction and thresholding
        binary_image = apply_background_subtraction(bg_image, gray_frame)

        # Update the background model
        bg_image = update_background(bg_image, gray_frame)

        # Display the resulting frame
        cv2.imshow('frame', binary_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
