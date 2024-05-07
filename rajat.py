import cv2
import mediapipe as mp
import numpy as np
import pytesseract
import threading
from collections import deque
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize virtual blackboard
board = None

# Initialize hand tracking variables
drawing = False
volume_control = False
erasing = False
prev_x, prev_y = 0, 0

# Set the path to the Tesseract executable (change to your Tesseract installation path)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize Tesseract configuration parameters
tess_config = r'--oem 3 --psm 6'

# Initialize deque to store recognized texts
recognized_texts = deque(maxlen=5)
consecutive_match = 0

# To access speaker through the library pycaw
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volMin, volMax = volume.GetVolumeRange()[:2]

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
                    elif erasing:
                        # Erase drawn lines
                        cv2.circle(board, (x, y), 20, (0, 0, 0), -1)
                    
                    # Update previous coordinates for smoother drawing
                    prev_x, prev_y = x, y
    
    return board

def clear_board():
    global board
    board = np.zeros_like(frame)

def recognize_text(roi):
    # Convert the region of interest to grayscale
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding if necessary
    _, thresh = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    # Perform OCR on the region of interest
    recognized_text = pytesseract.image_to_string(thresh, config=tess_config)
    
    return recognized_text

def print_recognized_text(roi):
    global consecutive_match
    recognized_text = recognize_text(roi)
    recognized_texts.append(recognized_text)
    if len(recognized_texts) == 5:
        if all(text == recognized_texts[0] for text in recognized_texts):
            if consecutive_match < 5:
                consecutive_match += 1
            else:
                print("Recognized text:", recognized_text)
        else:
            consecutive_match = 0

# Function to control volume
def control_volume(frame):
    global volume_control, prev_x, prev_y
    
    # Calculate hand position and adjust volume
    lmList = []  # empty list
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):  # adding counter and returning it
                # Get finger joint points
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])  # adding to the empty list 'lmList'
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        if lmList != []:
            # getting the value at a point
            # x      #y
            x1, y1 = lmList[4][1], lmList[4][2]  # thumb
            x2, y2 = lmList[8][1], lmList[8][2]  # index finger

            length = hypot(x2 - x1, y2 - y1)  # distance b/w tips using hypotenuse
            vol = np.interp(length, [30, 350], [volMin, volMax])
            volume.SetMasterVolumeLevel(vol, None)
            print("Volume:", vol)
            prev_x, prev_y = x2, y2

# Initialize webcam
cap = cv2.VideoCapture("http://192.168.1.7:4747/video")

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
            volume_control = False
            erasing = False
        elif key == ord('v'):
            # Toggle volume control mode
            volume_control = not volume_control
            drawing = False
            erasing = False
        elif key == ord('e'):
            # Toggle erasing mode
            erasing = not erasing
            drawing = False
            volume_control = False
        elif key == ord('r'):
            # Reset the blackboard
            clear_board()
        
        # Perform character recognition if drawing
        if drawing:
            roi = frame[prev_y:, prev_x:]  # Assuming prev_x, prev_y are the starting coordinates
            
            # Create a thread for printing recognized text
            recognition_thread = threading.Thread(target=print_recognized_text, args=(roi,))
            recognition_thread.start()
        
        # Perform volume control if enabled
        if volume_control:
            control_volume(frame)
        
    cap.release()
    cv2.destroyAllWindows()
