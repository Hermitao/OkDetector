import numpy as np
import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

p_time = 0
c_time = 0

last_4_position = [0, 0]
last_8_position = [0, 0]

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_lms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.putText(img, str(id), (cx, cy), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                #cv2.putText(img, "[{cx}, {cy}]".format(cx = cx, cy =cy ), (cx - 10, cy + 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)

                last_4_vector = np.array(last_4_position)
                last_8_vector = np.array(last_8_position)
                distance = np.linalg.norm(last_8_vector - last_4_vector)
                print(last_8_vector)

                if id == 4:
                    last_4_position = [lm.x, lm.y]
                    cv2.putText(img, "{distance}".format(distance=distance), (cx - 10, cy + 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
                if id == 8:
                    last_8_position = [lm.x, lm.y]
                    cv2.putText(img, "{distance}".format(distance=distance), (cx - 10, cy + 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)

                if distance < 0.1:
                    cv2.putText(img, "OK!", (10, 140), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)

            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

    c_time = time.time()
    fps = 1/(c_time - p_time)
    p_time = c_time

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)