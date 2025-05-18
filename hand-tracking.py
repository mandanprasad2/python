import cv2
import mediapipe as mp
import pyautogui
import time
import math

# Initialize MediaPipe Hands and drawing utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Screen size
screen_width, screen_height = pyautogui.size()

# Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1920)  # width
cap.set(4, 1020)  # height

# Smoothening params
prev_x, prev_y = 0, 0
smoothening = 7

# Click debounce
last_click_time = 0
click_delay = 0.5  # seconds

while True:
    success, frame = cap.read()
    if not success:
        break

    # Flip the frame horizontally for natural movement
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get landmark positions
            landmarks = hand_landmarks.landmark

            # Index finger tip (8) and thumb tip (4)
            index_tip = landmarks[8]
            thumb_tip = landmarks[4]

            # Convert to pixel coordinates
            index_x = int(index_tip.x * w)
            index_y = int(index_tip.y * h)
            thumb_x = int(thumb_tip.x * w)
            thumb_y = int(thumb_tip.y * h)

            # Print debug info
            print(f"Index: ({index_x}, {index_y}), Thumb: ({thumb_x}, {thumb_y})")

            # Convert index tip to screen coordinates
            screen_x = screen_width * index_tip.x
            screen_y = screen_height * index_tip.y

            # Smooth movement
            curr_x = prev_x + (screen_x - prev_x) / smoothening
            curr_y = prev_y + (screen_y - prev_y) / smoothening
            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

            # Draw circles to mark index and thumb tips
            cv2.circle(frame, (index_x, index_y), 10, (255, 0, 0), cv2.FILLED)  # Blue for index
            cv2.circle(frame, (thumb_x, thumb_y), 10, (0, 0, 255), cv2.FILLED)  # Red for thumb

            # Calculate distance between index and thumb for click
            distance = math.hypot(thumb_x - index_x, thumb_y - index_y)

            # Show distance on screen
            cv2.putText(frame, f'Distance: {int(distance)}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Click detection
            if distance < 100:
                if time.time() - last_click_time > click_delay:
                    pyautogui.click()
                    last_click_time = time.time()
                # Indicate click visually
                cv2.circle(frame, ((index_x + thumb_x) // 2, (index_y + thumb_y) // 2), 15, (0, 255, 0), cv2.FILLED)


    # Show the video
    cv2.imshow("Hand Tracking Mouse", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
