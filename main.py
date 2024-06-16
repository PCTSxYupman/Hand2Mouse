import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import tensorflow as tf

# Check available GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Allow memory growth for the GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Set TensorFlow to use GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        print("Using GPU:", gpus[0])
    except RuntimeError as e:
        print(e)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands.Hands(
    max_num_hands=1,                   # Limit to detecting 1 hand for higher frame rate
    min_detection_confidence=0.5,       # Confidence threshold to detect a hand
    min_tracking_confidence=0.5        # Confidence threshold to track a hand
)

# Screen resolution (adjust according to your screen resolution)
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

# Function to map coordinates from hand space to screen space and clamp to screen boundaries
def map_coordinates(x, y):
    screen_x = np.interp(x, [0, 1], [0, SCREEN_WIDTH])  # Adjusted with full screen width
    screen_y = np.interp(y, [0, 1], [0, SCREEN_HEIGHT])  # Adjusted with full screen height
    return int(screen_x), int(screen_y)

# Function to check if middle finger is not colliding with index finger
def are_fingers_apart(hand_landmarks):
    middle_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
    index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
    # Calculate Euclidean distance between middle and index finger tips
    distance = np.sqrt((middle_tip.x - index_tip.x)**2 + (middle_tip.y - index_tip.y)**2)
    # Define a threshold distance (adjust as needed)
    collision_threshold = 0.1  # Increased for larger hitbox
    return distance >= collision_threshold

def main():
    # Initialize OpenCV video capture for camera index 1
    cap = cv2.VideoCapture(1)

    # Set camera resolution (adjust as needed)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while cap.isOpened():
        # Read frame from the camera
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        # Convert the image from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image with MediaPipe Hands
        results = mp_hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get the coordinates of middle and index finger tips
                middle_x = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP].x
                middle_y = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP].y
                index_x = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].x
                index_y = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y

                # Map hand coordinates to screen coordinates
                screen_middle_x, screen_middle_y = map_coordinates(middle_x, middle_y)
                screen_index_x, screen_index_y = map_coordinates(index_x, index_y)

                # Draw larger circles around finger tips for visualization (hitbox)
                cv2.circle(frame, (screen_middle_x, screen_middle_y), 20, (0, 255, 0), -1)
                cv2.circle(frame, (screen_index_x, screen_index_y), 20, (0, 255, 0), -1)

                # Move the mouse cursor using the middle finger tip position
                pyautogui.moveTo(screen_middle_x, screen_middle_y, duration=0.1)

                # Check if middle finger is not colliding with index finger
                if are_fingers_apart(hand_landmarks):
                    # Perform click action
                    pyautogui.click()

        # Display the frame
        cv2.imshow('Hand Tracking', frame)

        # Exit the loop by pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
