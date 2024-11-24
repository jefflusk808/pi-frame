

'''
Testing information
camera = PiCamera()

camera.start_preview()
sleep(15)
camera.stop_preview()
'''
import cv2
import mediapipe as mp
from picamera2 import Picamera2
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize PiCamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

# FPS calculation variables
start_time = time.time()
frame_count = 0

def get_hand_gesture(landmarks):
    # Extract key points
    thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    
    # Simple gesture recognition logic
    if all(finger.y < landmarks.landmark[0].y for finger in [index_tip, middle_tip, ring_tip, pinky_tip]):
        return "Open Palm"
    elif all(finger.y > landmarks.landmark[0].y for finger in [index_tip, middle_tip, ring_tip, pinky_tip]):
        return "Closed Fist"
    elif index_tip.y < landmarks.landmark[0].y and middle_tip.y < landmarks.landmark[0].y:
        return "Peace Sign"
    elif thumb_tip.x > index_tip.x:
        return "Thumbs Up"
    else:
        return "Unknown"
    

while True:
    # Capture frame from PiCamera2
    frame = picam2.capture_array()

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    # Draw hand landmarks on the frame
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get hand gesture
            gesture = get_hand_gesture(hand_landmarks)
            cv2.putText(frame, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Calculate and display FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        frame_count = 0
        start_time = time.time()

    # Display the frame
    cv2.imshow('Hand Gesture Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
picam2.stop()
hands.close()