import cv2
import mediapipe as mp
import numpy as np
import csv
import os
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

record = False
class_label = 0
sequence_id = -1 
frame_in_seq = 0
filename = "gesture_data.csv"

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

header = ['seq_id', 'frame'] + \
         [f'{axis}{i}' for i in range(1, 22) for axis in ('x', 'y', 'z')] + ['label']

if os.path.exists(filename):
    with open(filename) as fr:
        rows = [r for r in csv.reader(fr) if r]
    sequence_id = max((int(r[0]) for r in rows[1:]), default=-1)  # continue numbering
else:
    with open(filename, 'w', newline='') as fw:
        csv.writer(fw).writerow(header)
    sequence_id = -1

f = open(filename, mode='a', newline='')   # append, never overwrite
writer = csv.writer(f)
frame_in_seq = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(RGB_frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):
        record = not record
        if record:
            sequence_id += 1
            frame_in_seq = 0
    if key == ord('n'):
        class_label += 1

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            if record:
                wrist = handLms.landmark[mp_hands.HandLandmark.WRIST]
                index_mcp = handLms.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                dist = np.linalg.norm(
                    np.array([index_mcp.x, index_mcp.y]) - np.array([wrist.x, wrist.y])
                )
                if dist == 0:
                    dist = 0.001

                hand_features = []
                for lm in handLms.landmark:
                    hand_features.append(float((lm.x - wrist.x) / dist))
                    hand_features.append(float((lm.y - wrist.y) / dist))
                    hand_features.append(float((lm.z - wrist.z) / dist))

                writer.writerow([sequence_id, frame_in_seq] + hand_features + [class_label])
                frame_in_seq += 1

    status = f"REC seq={sequence_id} f={frame_in_seq}" if record else "PAUSED"
    color = (0, 255, 0) if record else (0, 0, 255)
    cv2.putText(frame, f"{status}  label={class_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Frame", frame)
    if key == ord('q'):
        break

f.close()
cv2.destroyAllWindows()
cap.release()