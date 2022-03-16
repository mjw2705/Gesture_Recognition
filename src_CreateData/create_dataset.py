import cv2
import mediapipe as mp
import time, os
import numpy as np


def calc_angle(joint):
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
    return angle


actions = ['palm', 'quiet', 'grab', 'pinch']
dirs = 'dataset/ex_dataset'
os.makedirs(dirs, exist_ok=True)

seq_length = 30
secs_for_action = 100

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp.solutions.hands.Hands(static_image_mode=False,
                                max_num_hands=1,
                                min_detection_confidence=0.5,
                                min_tracking_confidence=0.5)

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

while cap.isOpened():
    for idx, action in enumerate(actions):
        data = []
        ret, img = cap.read()

        img = cv2.flip(img, 1)

        cv2.putText(img, f'Waiting for collecting {action.upper()} action...',
                    org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(255, 255, 255), thickness=2)
        cv2.imshow('img', img)
        cv2.waitKey(3000)

        start_time = time.time()

        while time.time() - start_time < secs_for_action:
            ret, img = cap.read()
            img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)

            results = hands.process(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    joint = np.zeros((21, 4))

                    for i, lm in enumerate(hand_landmarks.landmark):
                        joint[i] = [lm.x, lm.y, lm.z, lm.visibility]

                    angle = calc_angle(joint)

                    angle_label = np.array([angle], dtype=np.float32)
                    angle_label = np.append(angle_label, idx)

                    d = np.concatenate([joint.flatten(), angle_label])
                    data.append(d)

                    mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.imshow('img', img)
            if cv2.waitKey(1) == 27:
                break

        data = np.array(data)
        print(action, data.shape)
        np.save(os.path.join(dirs, f'raw_{action}'), data)

        # Create sequence data
        full_seq_data = []
        for seq in range(len(data) - seq_length):
            full_seq_data.append(data[seq:seq + seq_length])

        full_seq_data = np.array(full_seq_data)
        print(action, full_seq_data.shape)
        np.save(os.path.join(dir, f'seq_{action}'), full_seq_data)

    break