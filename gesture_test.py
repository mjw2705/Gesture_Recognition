# pyinstaller -D --icon=handicon.ico gesture.py
import csv
import time
import math
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import onnxruntime
import utils


# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hand = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

gesture_session = onnxruntime.InferenceSession('onnx/gesture_model.onnx', None)
face_session = onnxruntime.InferenceSession('onnx/facedetector.onnx', None)
input_name = face_session.get_inputs()[0].name
class_names = ["BACKGROUND", "FACE"]

actions = ['palm', 'quiet', 'grab', 'pinch']

# 변수
pTime = 0
swipe_seq = []
before_x = 0
before_y = 0
count = 0
seq_joint = []

frames = []

frame_w = 640
frame_h = 480

# 좌표값 저장
csv_path = 'point_history.csv'
with open(csv_path, 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Frame num','action', 'angle', 'x1,y1','x2,y2','x3,y3','x4,y4','x5,y5','x6,y6','x7,y7','x8,y8','x9,y9','x10,y10',
                     'x11,y11','x12,y12','x13,y13','x14,y14','x15,y15','x16,y16','x17,y17','x18,y18','x19,y19','x20,y20','x21,y21'])


# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)
cap.set(cv2.CAP_PROP_FPS, 120)

while cap.isOpened():
    ret, image = cap.read()

    image = cv2.flip(image, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hands = hand.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # face detect
    img1 = cv2.resize(image, (320, 240))
    image_mean = np.array([127, 127, 127])
    img1 = (img1 - image_mean) / 128
    img1 = np.transpose(img1, [2, 0, 1])
    img1 = np.expand_dims(img1, axis=0)
    img1 = img1.astype(np.float32)

    confidences, boxes = face_session.run(None, {input_name: img1})
    boxes, labels, probs = utils.predict(image.shape[1], image.shape[0], confidences, boxes, 0.9)
    try:
        boxe_width = [int(w[2] - w[0]) for w in boxes]
        box_idx = np.argmax(boxe_width)
        box = boxes[box_idx, :]
        # add box margin
        box_margin = int((box[2] - box[0]) / 2)
        box = [box[0] - box_margin, box[1] - box_margin, box[2] + box_margin, box[3] + box_margin]

        width = box[2] - box[0]
        height = box[3] - box[1]

        new_x1 = box[0] - width
        new_x2 = box[2] + width
        new_y2 = box[3] + height * 2

        # draw gesture region
        cv2.rectangle(image, (new_x1, box[1]), (new_x2, new_y2), (0, 0, 255), 1, cv2.LINE_AA)

        img_roi = image[box[1]:new_y2, new_x1:new_x2].copy()
        h, w, _ = img_roi.shape

        # hand
        if hands.multi_hand_landmarks:
            for idx, res in enumerate(hands.multi_hand_landmarks):
                label = hands.multi_handedness[idx].classification[0].label

                joint = np.zeros((21, 4))
                abs_joint = np.zeros((21, 2))

                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z, lm.visibility]
                    abs_joint[j] = [int(lm.x * frame_w), int(lm.y * frame_h)]

                seq_joint.append(joint)

                # 제스쳐 영역 내에서만 제스쳐 가능
                if np.any(abs_joint[:, 0] >= new_x1) and np.any(abs_joint[:, 0] <= new_x2) \
                        and np.any(abs_joint[:, 1] > box[1]) and np.any(abs_joint[:, 1] < new_y2):

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

                    mp_drawing.draw_landmarks(image, res, mp_hands.HAND_CONNECTIONS)

                    input_data = np.expand_dims(np.array(d, dtype=np.float32), axis=0)

                    input_data = input_data if isinstance(input_data, list) else [input_data]
                    feed = dict([(input.name, input_data[n]) for n, input in enumerate(gesture_session.get_inputs())])

                    y_pred = gesture_session.run(None, feed)[0].squeeze()
                    i_pred = int(np.argmax(y_pred))

                    conf = y_pred[i_pred]

                    if conf < 0.9:
                        continue

                    action = actions[i_pred]

                    # swipe 알고리즘
                    if action == 'swipe':
                        swipe_seq.append(action)

                        if len(swipe_seq) == 1:
                            before_x = joint[12][0]
                            before_y = joint[12][1]

                        if len(swipe_seq) >= 8:
                            if abs(before_x - joint[12][0]) >= 0.2:
                                if before_x > joint[12][0]:
                                    action = 'left'
                                    swipe_seq = []
                                elif before_x < joint[12][0]:
                                    action = 'right'
                                    swipe_seq = []
                            if abs(before_y - joint[12][1]) >= 0.2:
                                if before_y > joint[12][1]:
                                    action = 'up'
                                    swipe_seq = []
                                elif before_y < joint[12][1]:
                                    action = 'down'
                                    swipe_seq = []

                    # grab 각도 계산
                    if action == 'grab':
                        if label == 'Left':
                            angle2 = (math.degrees(math.atan2(joint[3][1] - joint[17][1],
                                                              joint[3][0] - joint[17][0])))
                        else:
                            angle2 = (math.degrees(math.atan2(joint[17][1] - joint[3][1],
                                                              joint[17][0] - joint[3][0])))

                        utils.draw_timeline(image, angle, abs_joint)

                    if action != 'swipe':
                        swipe_seq = []

                    print(count, action)
                    cv2.putText(image, f'{action.upper()}',
                                org=(int(abs_joint[0][0]), int(abs_joint[0][1] + 20)),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

                    if action == 'grab':
                        utils.logging_csv(csv_path, count, action, angle2, abs_joint)
                    else:
                        utils.logging_csv(csv_path, count, action, 'None', abs_joint)

                else:
                    continue

        else:
            # 보간
            if len(seq_joint) >= 2:
                new_joint = np.zeros((21, 3))
                for j1, j2 in zip(seq_joint[-2], seq_joint[-1]):
                    for idx in range(21):
                        x = (j1[idx][0] + j2[idx][0]) // 2
                        y = (j1[idx][1] + j2[idx][1]) // 2
                        z = (j1[idx][2] + j2[idx][2]) // 2
                        new_joint[idx] = [x, y, z]

                print(new_joint)
            else:
                continue

    except:
        print("Face detecting error!")


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    delay = 1000 / fps

    cv2.putText(image, f"FPS : {fps:.2f}", (10, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 1)
    cv2.imshow('img', image)

    count += 1
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()