import cv2
import mediapipe as mp
import numpy as np
import os
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

LABEL = '9'  # Change this to your current letter
DATA_PATH = f'data/{LABEL}'
os.makedirs(DATA_PATH, exist_ok=True)

cap = cv2.VideoCapture(0)
i = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    elapsed = int(time.time() - start_time)
    cv2.putText(frame, f"Time: {elapsed}s | Samples: {i}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Label: {LABEL} | Press 'q' to quit", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            np.save(os.path.join(DATA_PATH, f"{i}.npy"), np.array(landmarks))
            i += 1
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Collecting", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()