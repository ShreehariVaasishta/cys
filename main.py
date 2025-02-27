import cv2
import mediapipe as mp
import pyautogui

# Initialize video capture from the default webcam (index 0)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

# Get the screen size for mapping hand coordinates to screen coordinates
screen_width, screen_height = pyautogui.size()

# Initialize Mediapipe Hands solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# State variable to prevent multiple clicks during a single gesture
was_clicking = False

# Main loop
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Flip the frame horizontally for a mirror-like effect (optional)
    frame = cv2.flip(frame, 1)

    # Get frame dimensions
    frame_height, frame_width, _ = frame.shape

    # Convert the frame from BGR to RGB (Mediapipe requires RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    results = hands.process(frame_rgb)

    # If hands are detected, process gestures
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame for visual feedback
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get index finger tip coordinates (landmark 8)
            index_tip = hand_landmarks.landmark[8]
            index_x = int(index_tip.x * frame_width)
            index_y = int(index_tip.y * frame_height)

            # Get thumb tip coordinates (landmark 4)
            thumb_tip = hand_landmarks.landmark[4]
            thumb_x = int(thumb_tip.x * frame_width)
            thumb_y = int(thumb_tip.y * frame_height)

            # Calculate distance between thumb and index finger tips
            distance = ((index_x - thumb_x) ** 2 + (index_y - thumb_y) ** 2) ** 0.5

            # Perform a click if fingers are close (threshold: 30 pixels)
            if distance < 30:
                if not was_clicking:
                    pyautogui.click()
                    was_clicking = True
            else:
                was_clicking = False

            # Map index finger position to screen coordinates
            screen_x = int(index_x * screen_width / frame_width)
            screen_y = int(index_y * screen_height / frame_height)

            # Move the cursor to the mapped position
            pyautogui.moveTo(screen_x, screen_y)

    # Display the frame with hand landmarks
    cv2.imshow("Hand Detection", frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Clean up: release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
