# coding: utf-8
import os
import csv
import cv2
import numpy as np
import mediapipe as mp


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
    cv2.putText(img, text='Angle %d' % (angle2,), org=(50, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, color=(200, 10, 10), thickness=2)

def face_detect(image, face_session, input_name):
    face_img = cv2.resize(image, (320, 240))
    image_mean = np.array([127, 127, 127])
    face_img = (face_img - image_mean) / 128
    face_img = np.transpose(face_img, [2, 0, 1])
    face_img = np.expand_dims(face_img, axis=0)
    face_img = face_img.astype(np.float32)
    confidences, boxes = face_session.run(None, {input_name: face_img})
    boxes, labels, probs = predict(image.shape[1], image.shape[0], confidences, boxes, 0.9)

    return boxes,labels, probs

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


def area_of(left_top, right_bottom):
    """Compute the areas of rectangles given two corners.
    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.
    Returns:
        area (N): return the area.
    """
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.
    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """
    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
         picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    # _, indexes = scores.sort(descending=True)
    indexes = np.argsort(scores)
    # indexes = indexes[:candidate_size]
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        # current = indexes[0]
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        # indexes = indexes[1:]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]


def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = hard_nms(box_probs,
                             iou_threshold=iou_threshold,
                             top_k=top_k,
                             )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]