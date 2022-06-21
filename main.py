import numpy as np
import cv2
import mediapipe as mp
import time
import tensorflow as tf
import pandas as pd
import warnings as CALA_BOCA
CALA_BOCA.filterwarnings('ignore')

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

p_time = 0
c_time = 0

base_de_dados = pd.DataFrame()
for x in range(21):
    base_de_dados[str(x) + '_x'] = pd.Series([])
    base_de_dados[str(x) + '_y'] = pd.Series([])
        
base_de_dados['Classe'] = ''

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    h, w, c = img.shape

    nova_linha = ['' for _ in range(42)] 

    if results.multi_hand_landmarks:

        positions = [0, 0]

        for hand_lms in results.multi_hand_landmarks:

            ref_lm = hand_lms.landmark[17]
            ref_position = np.array([ref_lm.x, ref_lm.y])

            origin_lm = hand_lms.landmark[0]
            origin_position = np.array([origin_lm.x, origin_lm.y])

            ref_origin_distance = np.linalg.norm(ref_position - origin_position)

            cv2.putText(img, f"{ref_origin_distance:.1f}", (int(ref_lm.x * w), int(ref_lm.y * h)), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

            # for n in range(0, 42, 2): 
            #     positiony_relative = hand_lms.landmark[n][1].y - hand_lms.landmark[n].y
            #     positionx_relative = hand_lms.landmark[n][1].x - hand_lms.landmark[n].x

            #     positionx_relative_normalized = positionx_relative / ref_origin_distance
            #     positiony_relative_normalized = positiony_relative / ref_origin_distance

            #     nova_linha[n] = positionx_relative_normalized
            #     nova_linha[n + 1] = positiony_relative_normalized

            for id, lm in enumerate(hand_lms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)

                positiony_relative = lm.y - hand_lms.landmark[0].y
                positionx_relative = lm.x - hand_lms.landmark[0].x

                positionx_relative_normalized = positionx_relative / ref_origin_distance
                positiony_relative_normalized = positiony_relative / ref_origin_distance

                nova_linha[id] = positionx_relative_normalized
                nova_linha[id + 1] = positiony_relative_normalized

                my_position = np.array([lm.x, lm.y])
                distance = np.linalg.norm(my_position - origin_position)
                distance_normalized = (distance) / ref_origin_distance

                # cv2.putText(img, f"{positionx_relative_normalized:.2f}     {positiony_relative_normalized:.2f}", (cx, cy), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                # cv2.putText(img, f"{distance_normalized:.2f}", (cx, cy), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                # cv2.putText(img, f"{distance:.2f}", (cx, cy + 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 122, 200), 1)
                # cv2.putText(img, f"{lm.x:.2f}   {lm.y:.2f}", (cx, cy), cv2.FONT_HERSHEY_PLAIN, 1, (0, 122, 200), 1)
                cv2.putText(img, f"{positionx_relative_normalized:.2f}   {positiony_relative_normalized:.2f}", (cx, cy), cv2.FONT_HERSHEY_PLAIN, 1, (0, 122, 200), 1)

            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

    c_time = time.time()
    fps = 1/(c_time - p_time)
    p_time = c_time

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('a'):
        base_de_dados = base_de_dados.append(nova_linha)
        print ("aaaaaaaaaaaaaaaaaaaaaaaaa")
    if cv2.waitKey(1) == ord('e'):
        base_de_dados.to_excel(r'C:\Users\Aluno\Desktop\OkDetector-libras\baseDeDados.xlsx', index=False)
        break
