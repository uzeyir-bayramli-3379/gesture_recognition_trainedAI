import cv2
import mediapipe as mp
import numpy as np 
import torch
import torch.nn as nn
from PIL import Image, ImageSequence

#gif = Image.open("cat-scuba-cat.gif")
#gif_frames = []
#for frame in ImageSequence.Iterator(gif):
#    f = frame.convert("RGB")
#    f = cv2.cvtColor(np.array(f), cv2.COLOR_RGB2BGR)
#    gif_frames.append(f)
#gif_idx = 0
#gif_window_open = False


is_pose_0=False
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)
class CNN_model(nn.Module):
    def __init__(self):
        super(CNN_model, self).__init__()
        self.layer1 = nn.Conv1d(in_channels=63, out_channels=32, kernel_size=5, stride=2)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3)
        self.act2 = nn.ReLU()
        self.m=nn.AdaptiveAvgPool1d(1)
        self.layer3=nn.Linear(32,3)
    def forward(self, x):
        x = self.layer1(x)
        x = self.act1(x)
        x = self.layer2(x)
        x= self.act2(x)
        x= self.m(x)
        x=x.squeeze(-1)
        x=self.layer3(x)
        return x
model=CNN_model()
model.load_state_dict(torch.load('scuba_model.pth', weights_only=True))
model.eval()
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
buffer=[]
while True:
    success, frame = cap.read()
    if not success:
        break
    is_pose_0=False
    RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(RGB_frame)
    key = cv2.waitKey(1) & 0xFF

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            wrist = handLms.landmark[mp_hands.HandLandmark.WRIST]
            index_mcp = handLms.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            
            dist = np.linalg.norm(np.array([index_mcp.x, index_mcp.y]) - np.array([wrist.x, wrist.y]))
            if dist == 0: 
                dist = 0.001
            hand_features = []
            for id, lm in enumerate(handLms.landmark):
                lm_x = float((lm.x - wrist.x) / dist)
                lm_y = float((lm.y - wrist.y) / dist)
                lm_z = float((lm.z - wrist.z) / dist)
                
                hand_features.append([lm_x, lm_y, lm_z])
            hand_features_flat = [coord for point in hand_features for coord in point]
            buffer.append(hand_features_flat)
            if len(buffer)>30:
                buffer.pop(0)
            if len(buffer)==30:
                window = np.array(buffer)
                window=np.transpose(window,(1,0))
                features_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    outputs = model(features_tensor)
                    predicted_class = torch.argmax(outputs, dim=1).item()
                    probs = torch.softmax(outputs, dim=1)
                    confidence = probs[0][predicted_class].item()
                if confidence > 0.6 and predicted_class == 0:
                    is_pose_0 = True

                cv2.putText(frame, f'Class: {predicted_class} ({confidence:.2f})', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    #if is_pose_0:
    #    cv2.imshow("Gesture GIF", gif_frames[gif_idx])
    #    gif_idx = (gif_idx + 1) % len(gif_frames)
    #    gif_window_open = True
    #elif gif_window_open:
    #    cv2.destroyWindow("Gesture GIF")
    #    gif_window_open = False

    cv2.imshow("Frame", frame)
    
    if key == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
