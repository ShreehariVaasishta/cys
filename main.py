import cv2
import mediapipe as mp
import pyautogui
import math

# Constants
THRESHOLD_MULTIPLIER = 0.3  # Click threshold as a fraction of hand size
N_STILL = 3  # Frames to check stillness
MOVEMENT_THRESHOLD = 0.05  # Max movement for stillness (normalized)
N_SMOOTH = 3  # Frames for smoothing cursor

# Initialize video capture and Mediapipe
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
screen_width, screen_height = pyautogui.size()

# State variables
was_clicking = False
wrist_history = []  # Wrist positions for stillness
position_history = []  # Cursor positions for smoothing

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Key landmarks
            wrist = hand_landmarks.landmark[0]
            middle_base = hand_landmarks.landmark[9]
            index_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]

            # Normalized distances
            ref_distance = math.hypot(wrist.x - middle_base.x, wrist.y - middle_base.y)
            distance_norm = math.hypot(
                index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y
            )
            threshold = ref_distance * THRESHOLD_MULTIPLIER

            # Stillness check
            wrist_history.append((wrist.x, wrist.y))
            if len(wrist_history) > N_STILL:
                wrist_history.pop(0)
            total_movement = (
                float("inf")
                if len(wrist_history) < N_STILL
                else sum(
                    math.hypot(
                        wrist_history[i][0] - wrist_history[i - 1][0],
                        wrist_history[i][1] - wrist_history[i - 1][1],
                    )
                    for i in range(1, N_STILL)
                )
            )

            # Click logic
            if (
                distance_norm < threshold
                and total_movement < MOVEMENT_THRESHOLD
                and not was_clicking
                and len(wrist_history) >= N_STILL
            ):
                pyautogui.click()
                was_clicking = True
            elif distance_norm >= threshold:
                was_clicking = False

            # Smooth cursor movement
            screen_x, screen_y = index_tip.x * screen_width, index_tip.y * screen_height
            position_history.append((screen_x, screen_y))
            if len(position_history) > N_SMOOTH:
                position_history.pop(0)
            avg_x = sum(p[0] for p in position_history) / len(position_history)
            avg_y = sum(p[1] for p in position_history) / len(position_history)
            pyautogui.moveTo(avg_x, avg_y)

    else:
        wrist_history.clear()
        position_history.clear()

    cv2.imshow("Hand Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
