import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hand Tracking
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Function to detect hand gestures
def detect_gestures(frame):
    landmarks = []
    hand_gestures = []

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                landmarks.append((x, y))

            # Draw hand landmarks
            mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)

            # Perform gesture recognition
            # You can implement gesture recognition logic here
            # For example, check if certain landmarks correspond to specific gestures
            # and append the detected gesture to hand_gestures list

    return frame, hand_gestures

# Function to display ATM interface
def display_atm_interface(frame):
    # You can implement the ATM interface here
    # This can include displaying buttons for different operations
    # such as withdrawing money, checking balance, etc.
    # Use OpenCV drawing functions to draw the interface elements on the frame

    return frame

# Main function
def main():
    # Open webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Detect gestures
        processed_frame, hand_gestures = detect_gestures(frame)

        # Display ATM interface
        processed_frame = display_atm_interface(processed_frame)

        # Display frame
        cv2.imshow("Virtual ATM", processed_frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
