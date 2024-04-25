import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize virtual blackboard
board = None

# Initialize hand tracking variables
drawing = False
prev_x, prev_y = 0, 0

def detect_hand_gestures(frame):
    global prev_x, prev_y, drawing, board
    
    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect hand landmarks
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for i, landmark in enumerate(hand_landmarks.landmark):
                h, w, _ = frame.shape
                x, y = int(landmark.x * w), int(landmark.y * h)
                if i == 8:  # Index fingertip
                    if drawing:
                        # Draw line between previous and current coordinates
                        cv2.line(board, (prev_x, prev_y), (x, y), (255, 255, 255), 5)
                    
                    # Update previous coordinates for smoother drawing
                    prev_x, prev_y = x, y
    
    return board

def clear_board():
    global board
    board = np.zeros_like(frame)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Set up media pipe hands
with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Create a black image as the virtual blackboard
        if board is None:
            board = np.zeros_like(frame)
        
        # Detect hand gestures and update virtual blackboard
        board = detect_hand_gestures(frame)
        
        # Display the webcam feed and the virtual blackboard
        cv2.imshow('Webcam', frame)
        cv2.imshow('Virtual Blackboard', board)
        
        # Check for key press events
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('d'):
            # Toggle drawing mode
            drawing = not drawing
        elif key == ord('e'):
            # Erase mode
            drawing = True
            clear_board()

    cap.release()
    cv2.destroyAllWindows()
