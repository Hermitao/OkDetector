import numpy as np
import cv2
import mediapipe as mp
import time
import tensorflow as tf

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
    # print(imgRGB.shape)
    results = hands.process(imgRGB)

    h, w, c = img.shape

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:

            ref_lm = hand_lms.landmark[17]
            ref_position = np.array([ref_lm.x, ref_lm.y])

            origin_lm = hand_lms.landmark[0]
            origin_position = np.array([origin_lm.x, origin_lm.y])

            ref_origin_distance = np.linalg.norm(ref_position - origin_position)
            print(ref_origin_distance)

            cv2.putText(img, f"{ref_origin_distance:.1f}", (int(ref_lm.x * w), int(ref_lm.y * h)), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

            for id, lm in enumerate(hand_lms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)

                positionx_relative = lm.x - hand_lms.landmark[0].x
                positiony_relative = lm.y - hand_lms.landmark[0].y

                positionx_relative_normalized = positionx_relative / ref_origin_distance
                positiony_relative_normalized = positiony_relative / ref_origin_distance

                my_position = np.array([lm.x, lm.y])
                distance = np.linalg.norm(my_position - origin_position)
                distance_normalized = (distance) / ref_origin_distance

                # cv2.putText(img, f"{positionx_relative_normalized:.2f}     {positiony_relative_normalized:.2f}", (cx, cy), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                cv2.putText(img, f"{distance_normalized:.1f}", (cx, cy), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                cv2.putText(img, f"{distance:.1f}", (cx, cy + 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 122, 200), 1)

            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

    c_time = time.time()
    fps = 1/(c_time - p_time)
    p_time = c_time

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
