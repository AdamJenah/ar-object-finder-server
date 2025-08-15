import cv2
import numpy as np

# NOTE: You must adapt these functions based on your yolov11n-pose ONNX outputs.
# These are stubs showing the expected interface used by main.py.

def preprocess_pose_input(image_bgr, size=640):
    img = cv2.resize(image_bgr, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None, ...]  # NCHW
    return img, size

def run_pose(session, bgr_img):
    # Example only: replace with your actual postprocess to extract keypoints
    # outputs = session.run(None, {session.get_inputs()[0].name: inp})
    # parse outputs -> list of dicts, each with keypoints as (x,y) in original image coords
    # e.g., [{"left_shoulder":(x,y), "right_shoulder":(x,y), "left_hip":(x,y), ...}, ...]
    return []

def clothing_rois_from_keypoints(people, target="shirt"):
    rois = []
    for p in people:
        ls = p.get("left_shoulder"); rs = p.get("right_shoulder")
        lh = p.get("left_hip");      rh = p.get("right_hip")
        lk = p.get("left_knee");     rk = p.get("right_knee")
        if target == "shirt" and ls and rs and lh and rh:
            x1 = int(min(ls[0], rs[0], lh[0], rh[0])); x2 = int(max(ls[0], rs[0], lh[0], rh[0]))
            y1 = int(min(ls[1], rs[1]));              y2 = int(max(lh[1], rh[1]))
            rois.append([max(0,x1), max(0,y1), max(0,x2), max(0,y2)])
        if target == "pants" and lh and rh and (lk or rk):
            knees_y = np.median([v[1] for v in [lk, rk] if v is not None])
            x1 = int(min(lh[0], rh[0])); x2 = int(max(lh[0], rh[0]))
            y1 = int(min(lh[1], rh[1])); y2 = int(knees_y)
            rois.append([max(0,x1), max(0,y1), max(0,x2), max(0,y2)])
    return rois
