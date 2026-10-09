import cv2
import mediapipe as mp
import math
import numpy as np
import pyautogui

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

pyautogui.FAILSAFE = False

screen_w, screen_h = pyautogui.size()

frameR = 50
smoothening = 3

prev_x, prev_y = 0, 0
clicking = False

cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    x1 = y1 = x2 = y2 = None

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(img_rgb)
    

    if results.multi_hand_landmarks:

        h, w, _ = frame.shape

        for handLms in results.multi_hand_landmarks:

            for id, Lm in enumerate(handLms.landmark):

                cx, cy = int(Lm.x * w), int(Lm.y * h)

                if id == 4:
                    x1, y1 = cx, cy

                if id == 8:
                    x2, y2 = cx, cy

            

            if None not in (x1, y1, x2, y2):

                # Adaptive threshold
                if y2 < 0.6 * h:
                    threshold = 43
                else:
                    threshold = 52

                length = math.hypot(x2 - x1, y2 - y1)

                if length < threshold:

                    if not clicking:
                        pyautogui.click()
                        clicking = True

                    cv2.circle(frame, (x2, y2), 15,
                               (180, 142, 89), cv2.FILLED)

                else:

                    clicking = False

                    cv2.circle(frame, (x2, y2), 15,
                               (255, 0, 0), cv2.FILLED)

            

            if x2 is not None and y2 is not None:

                mapped_x = np.interp(
                    x2,
                    (frameR, w - frameR),
                    (0, screen_w)
                )

                mapped_y = np.interp(
                    y2,
                    (frameR, h - 20),
                    (0, screen_h - 80)
                )

                # Smooth movement
                x = prev_x + (mapped_x - prev_x) / smoothening
                y = prev_y + (mapped_y - prev_y) / smoothening

                # Screen boundary clamp
                x = max(0, min(screen_w, x))
                y = max(0, min(screen_h - 80, y))

                # Move cursor only when not clicking
                if not clicking:
                    pyautogui.moveTo(int(x), int(y))

                # Update internal smoothening state
                prev_x, prev_y = x, y

                # Debug y2 value
                cv2.putText(
                    frame,
                    f"y2: {y2}",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2
                )

            mp_draw.draw_landmarks(
                frame,
                handLms,
                mp_hands.HAND_CONNECTIONS
            )

            cv2.rectangle(frame,(frameR, frameR),(w - frameR, h - frameR),(255, 0, 0),2)

    cv2.imshow("Air Mouse", frame)

    if cv2.waitKey(1) & 0xFF == ord('p'):
        break

cap.release()
cv2.destroyAllWindows()