import math
import sys
import time
import cv2
import numpy as np
import onnxruntime
import mediapipe as mp

import utils



gesture_session = onnxruntime.InferenceSession('gesture_model2.onnx', None)


mp_holistic = mp.solutions.holistic
holistic = mp.solutions.holistic.Holistic(static_image_mode=False,
                                          min_detection_confidence=0.6,
                                          min_tracking_confidence=0.5)

actions = ['palm', 'quiet', 'grab', 'pinch', 'None']

frame_w = 640
frame_h = 480
pTime = 0
hand_center_x = 0
hand_center_y = 0
face_sp, face_ep = 0, 0
quiet_y1, quiet_y2 = 0, 0
joint = None
abs_joint = None
label = None

palm_xlocs = []
palm_ylocs = []
swipe_q = ['none']
swipe_val = 'none'

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)
cap.set(cv2.CAP_PROP_FPS, 30)

while cap.isOpened():
    ret, image = cap.read()
    image = cv2.flip(image, 1)

    if not ret:
        print("Ignoring empty camera frame.")
        continue

    image.flags.writeable = False
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb)
    image.flags.writeable = True

    if results.face_landmarks is None:
        print('No face')
        action = actions[-1]

    else:
        pose_lms = [[landmark.x, landmark.y, landmark.z, landmark.visibility]
                    for landmark in results.pose_landmarks.landmark]

        face_sp, face_ep = utils.pose_face(pose_lms, frame_w, frame_h)

        h = (face_ep[1] - face_sp[1]) // 2
        quiet_y1 = face_sp[1] + h
        quiet_y2 = face_ep[1] + (h // 2)

        if results.left_hand_landmarks:
            joint = np.zeros((21, 4))
            abs_joint = np.zeros((21, 2))

            for j, lm in enumerate(results.left_hand_landmarks.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]
                abs_joint[j] = [int(lm.x * frame_w), int(lm.y * frame_h)]
            label = 'Left'

        elif results.right_hand_landmarks:
            joint = np.zeros((21, 4))
            abs_joint = np.zeros((21, 2))

            for j, lm in enumerate(results.right_hand_landmarks.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]
                abs_joint[j] = [int(lm.x * frame_w), int(lm.y * frame_h)]
            label = 'Right'
        else:
            label = 'none'
            joint = None
            abs_joint = None


        if joint is not None:
            # 손바닥 중심 좌표
            if label == 'Right':
                hand_center_x = int(((pose_lms[16][0] + pose_lms[18][0] + pose_lms[20][0]) * frame_w) / 3)
                hand_center_y = int(((pose_lms[16][1] + pose_lms[18][1] + pose_lms[20][1]) * frame_h) / 3)

            elif label == 'Left':
                hand_center_x = int(((pose_lms[15][0] + pose_lms[17][0] + pose_lms[19][0]) * frame_w) / 3)
                hand_center_y = int(((pose_lms[15][1] + pose_lms[17][1] + pose_lms[19][1]) * frame_h) / 3)

            angle = utils.calc_angle(joint)
            d = np.concatenate([joint.flatten(), angle])

            conf, i_pred = utils.calc_predict(d, gesture_session)
            if conf < 0.9:
                continue

            action = actions[i_pred]

            # grab 각도 계산
            if action == 'grab':
                if label == 'Right':
                    grab_angle = math.degrees(math.atan2(joint[3][1] - joint[17][1],
                                                         joint[3][0] - joint[17][0]))

                else:
                    grab_angle = math.degrees(math.atan2(joint[17][1] - joint[3][1],
                                                         joint[17][0] - joint[3][0]))

                utils.draw_timeline(image, grab_angle, abs_joint)

            # quiet 영역 지정
            if action == 'quiet':
                if (face_sp[0] < abs_joint[7][0] < face_ep[0]) and (quiet_y1 < abs_joint[7][1] < quiet_y2):
                    action = 'quiet'
                else:
                    action = 'none'

            # swipe 알고리즘
            if action == 'palm':
                scaled_x = float(hand_center_x / frame_w)
                scaled_y = float(hand_center_y / frame_h)

                palm_xlocs.append(scaled_x)
                palm_ylocs.append(scaled_y)

                if len(palm_xlocs) > 5:
                    palm_xlocs.pop(0)
                if len(palm_ylocs) > 5:
                    palm_ylocs.pop(0)

                if len(palm_xlocs) == 5 and len(palm_ylocs) == 5:
                    diff_xlocs = [i - palm_xlocs[0] for i in palm_xlocs]
                    diff_ylocs = [i - palm_ylocs[0] for i in palm_ylocs]

                    if len(swipe_q) == 1:
                        if np.sum(diff_xlocs) > 0.2 and -0.3 < np.sum(diff_ylocs) < 0.3:
                            swipe_val = 'right'
                            swipe_q = utils.swipe(swipe_val)
                        elif np.sum(diff_xlocs) < -0.2 and -0.3 <  np.sum(diff_ylocs) < 0.3:
                            swipe_val = 'left'
                            swipe_q = utils.swipe(swipe_val)
                        elif np.sum(diff_ylocs) < -0.4:
                            swipe_val = 'up'
                            swipe_q = utils.swipe(swipe_val)
                        elif np.sum(diff_ylocs) > 0.2 and -0.5 < np.sum(diff_xlocs) < 0.5:
                            swipe_val = 'down'
                            swipe_q = utils.swipe(swipe_val)
            else:
                palm_xlocs.clear()
                palm_ylocs.clear()

            if len(swipe_q) > 1:
                swipe_val = swipe_q.pop()

            if action == 'palm':
                if swipe_val in ['left', 'right', 'up', 'down']:
                    action = swipe_val


        else:
            hand_center_x = 0
            hand_center_y = 0
            action = actions[-1]

    swipe_val = 'none'

    # display
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    if joint is not None:
        for lm in abs_joint:
            cv2.circle(image, (int(lm[0]), int(lm[1])), 2, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.circle(image, (hand_center_x, hand_center_y), 5, (255, 255, 0), -1, cv2.LINE_AA)
    # cv2.rectangle(image, (sx, sy), (ex, ey), (0, 0, 255), 1, cv2.LINE_AA)
    cv2.rectangle(image, (face_sp[0], quiet_y1), (face_ep[0], quiet_y2), (0, 0, 255))
    cv2.putText(image, f"FPS : {fps:.2f}", (10, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 1)
    cv2.putText(image, f'{action.upper()}',
                org=(510, 450),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=2)

    cv2.imshow('img', image)

    if cv2.waitKey(1) == 27:
        break


cap.release()
cv2.destroyAllWindows()