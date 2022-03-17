# coding: utf-8
import os
import csv
import cv2
import numpy as np
import mediapipe as mp


def get_landmark(landmarks, image):
    frame_h, frame_w = image.shape[:2]

    joint = np.zeros((21, 4))
    abs_joint = np.zeros((21, 2))

    for j, lm in enumerate(landmarks.landmark):
        joint[j] = [lm.x, lm.y, lm.z, lm.visibility]
        abs_joint[j] = [int(lm.x * frame_w), int(lm.y * frame_h)]

    return joint, abs_joint

def calc_predict(d, gesture_session):
    input_data = np.expand_dims(np.array(d, dtype=np.float32), axis=0)

    input_data = input_data if isinstance(input_data, list) else [input_data]
    feed = dict([(input.name, input_data[n]) for n, input in enumerate(gesture_session.get_inputs())])

    y_pred = gesture_session.run(None, feed)[0].squeeze()
    i_pred = int(np.argmax(y_pred))
    conf = y_pred[i_pred]

    return conf, i_pred


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

def logging_csv(csv_path, frame_num,  action, angle2, abs_joint):
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([frame_num, action, angle2, *abs_joint])
    return

def save_dict(hand_data, abs_joint, action, grab_angle):
    hand_data["devceid"] = 1
    hand_data["coordinate"] = [abs_joint[9][0], abs_joint[9][1]]  # 가운데 좌표
    hand_data["gesture"] = action
    hand_data["grab_angle"] = grab_angle


def draw_timeline(img, rel_x, joint):
    angle2 = rel_x
    img_h, img_w, _ = img.shape
    img_w /= 180

    timeline_w = int(img_w * rel_x) - 50

    cv2.rectangle(img, pt1=(int(img_w/2), img_h - 50), pt2=(int(img_w/2)+timeline_w, img_h - 48), color=(0, 0, 255), thickness=-1)
    cv2.line(img, (int(joint[3][0]), int(joint[3][1])), (int(joint[17][0]), int(joint[17][1])), (0, 0, 0), 4)
    cv2.putText(img, text='Angle %d' % (angle2,), org=(30, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.9, color=(200, 10, 10), thickness=2)


def box_pos(box, frame_w, frame_h):
    box_margin = int((box[2] - box[0]) / 2)
    box = [box[0] - box_margin, box[1] - box_margin, box[2] + box_margin, box[3] + box_margin]
    width = box[2] - box[0]
    height = box[3] - box[1]

    sx = max(box[0] - int(1.5 * width), 0)
    sy = max(box[1] - height // 2, 0)
    ex = min(box[2] + int(1.5 * width), frame_w)
    ey = min(box[3] + height, frame_h)

    return sx, sy, ex, ey

def pose_face(pose_lms, img_width, img_height):
    cx, cy = pose_lms[0][0], pose_lms[0][1]
    br = pose_lms[7][0]
    bl = pose_lms[8][0]
    bu = pose_lms[1][1]
    bb = pose_lms[10][1]

    w = max(br, bl) - min(br, bl)
    h = (max(bu, bb) - min(bu, bb)) * 2

    sx = int((cx - w / 2) * img_width)
    sy = int((cy - h / 2) * img_height)
    ex = sx + int(w * img_width)
    ey = sy + int(h * img_height)

    return (max(sx, 0), max(sy, 0)), (min(ex, img_width), min(ey, img_height))

def swipe(swipe_val):
    if swipe_val == 'right':
        swipe_q = ['right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right',
                    'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right',
                    'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right']
    elif swipe_val == 'left':
        swipe_q = ['left', 'left', 'left', 'left', 'left', 'left', 'left', 'left', 'left', 'left',
                    'left', 'left', 'left', 'left', 'left', 'left', 'left', 'left', 'left', 'left',
                    'left', 'left', 'left', 'left', 'left', 'left', 'left', 'left', 'left', 'left']
    elif swipe_val == 'up':
        swipe_q = ['up', 'up', 'up', 'up', 'up', 'up', 'up', 'up', 'up', 'up',
                    'up', 'up', 'up', 'up', 'up', 'up', 'up', 'up', 'up', 'up',
                    'up', 'up', 'up', 'up', 'up', 'up', 'up', 'up', 'up', 'up']
    elif swipe_val == 'down':
        swipe_q = ['down', 'down', 'down', 'down', 'down', 'down', 'down', 'down', 'down', 'down',
                    'down', 'down', 'down', 'down', 'down', 'down', 'down', 'down', 'down', 'down',
                    'down', 'down', 'down', 'down', 'down', 'down', 'down', 'down', 'down', 'down']

    return swipe_q


# -------------------------------not use---------------------------------- #
def box_poses(boxes):
    boxe_width = [int(w[2] - w[0]) for w in boxes]
    box_idx = np.argmax(boxe_width)
    box = boxes[box_idx, :]
    # add box margin
    box_margin = int((box[2] - box[0]) / 2)
    box = [box[0] - box_margin, box[1] - box_margin, box[2] + box_margin, box[3] + box_margin]
    width = box[2] - box[0]
    height = box[3] - box[1]

    sx = box[0] - width
    sy = box[1]
    ex = box[2] + width
    ey = box[3] + height

    return box, sx, sy, ex, ey

def mkdir(d):
    os.makedirs(d, exist_ok=True)

def hand(image, offset):
    hand =  mp.solutions.hands.Hands(static_image_mode=False,
                                    max_num_hands=1,
                                    min_detection_confidence=0.5,
                                    min_tracking_confidence=0.5)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape
    results = hand.process(image)

    if results.multi_hand_landmarks:
        res = results.multi_hand_landmarks[0]
        label = results.multi_handedness[0].classification[0].label
        joint = np.zeros((21, 4))
        abs_joint = np.zeros((21, 2))

        for j, lm in enumerate(res.landmark):
            joint[j] = [lm.x, lm.y, lm.z, lm.visibility]
            abs_joint[j] = [int(lm.x * w + offset[0]), int(lm.y * h + offset[1])]
        return joint, abs_joint, label
    return None, None, None