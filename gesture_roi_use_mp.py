import time

import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model



def set_res(cap, x,y):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(x))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(y))
    return str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hand = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Pose
mp_pose = mp.solutions.pose

actions = ['palm', 'quiet', 'grab', 'pinch']
seq_length = 30

model = load_model('models/raw_data_loss.h5')

# 변수
pTime = 0
cTime = 0
seq = []
action_seq = []

frame_w = 640
frame_h = 480
box_ratio = frame_h * 1.12

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
set_res(cap, frame_w, frame_h)


fourcc = cv2.VideoWriter_fourcc(*'DIVX')
# fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, img = cap.read()

        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hands = hand.process(img)
        poses = pose.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # pose
        pose_lms = [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in
                    poses.pose_landmarks.landmark]

        if np.shape(pose_lms) == (33, 4):
            # Face ROI define
            box_center = int(frame_w * (pose_lms[7][0] + pose_lms[8][0]) / 2)
            box_u = pose_lms[1][1]
            box_b = pose_lms[10][1]
            box_h = int(box_ratio * (box_b - box_u))
            y1 = int(frame_h * box_u - box_h)
            y2 = int(frame_h * box_b + box_h)
            x1 = int(box_center - 1.5 * box_h)
            x2 = int(box_center + 1.5 * box_h)

            width = x2 - x1
            height = y2 - y1

            new_x1 = x1 - width
            new_x2 = x2 + width
            new_y2 = y2 + height * 2

        img_copy = img[y1:new_y2, new_x1:new_x2].copy()

        # Draw face box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.rectangle(img, (new_x1, y1), (new_x2, new_y2), (0, 0, 255), 1, cv2.LINE_AA)

        # hand
        if hands.multi_hand_landmarks:
            for res in hands.multi_hand_landmarks:
                joint = np.zeros((21, 4))
                abs_joint = np.zeros((21, 2))

                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z, lm.visibility]
                    abs_joint[j] = [int(lm.x * frame_w), int(lm.y * frame_h)]


                if np.any(abs_joint[:, 0] >= new_x1) and np.any(abs_joint[:, 0] <= new_x2) \
                        and np.any(abs_joint[:, 1] > y1) and np.any(abs_joint[:, 1] < new_y2):

                    # Compute angles between joints
                    v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]  # Parent joint
                    v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]  # Child joint
                    v = v2 - v1  # [20, 3]
                    # Normalize v
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    # Get angle using arcos of dot product
                    angle = np.arccos(np.einsum('nt,nt->n',
                                                v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                                v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]

                    angle = np.degrees(angle)  # Convert radian to degree

                    d = np.concatenate([joint.flatten(), angle])

                    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

                    input_data = np.expand_dims(np.array(d, dtype=np.float32), axis=0)

                    y_pred = model.predict(input_data).squeeze()
                    i_pred = int(np.argmax(y_pred))
                    conf = y_pred[i_pred]

                    if conf < 0.9:
                        continue

                    action = actions[i_pred]

                    print(action)
                    cv2.putText(img, f'{action.upper()}',
                                org=(int(abs_joint[0][0]), int(abs_joint[0][1] + 20)),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)


        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 0, 0), 2)
        cv2.imshow('img', img)

        if cv2.waitKey(5) == 27:
            break

cap.release()
cv2.destroyAllWindows()