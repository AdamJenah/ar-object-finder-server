# utils.py
import cv2
import numpy as np

# Default COCO-80 names (edit if you trained custom classes)
COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
    "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
]

# --------- Preprocess (letterbox) ---------
def preprocess_bgr_to_yolo_input(bgr, size=640):
    """
    Letterbox an image to (size,size), return:
    - x: NCHW float32 [0,1] RGB input (shape (1,3,size,size))
    - r: scale factor used
    - dw, dh: padding offsets
    """
    h0, w0 = bgr.shape[:2]
    r = min(size / h0, size / w0)
    new_w, new_h = int(round(w0 * r)), int(round(h0 * r))
    resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    dw = (size - new_w) // 2
    dh = (size - new_h) // 2
    canvas[dh:dh + new_h, dw:dw + new_w] = resized

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    x = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]  # (1,3,H,W)
    return x, r, dw, dh

# --------- Helpers for postprocess ---------
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def _iou(a, bs):
    ax1, ay1, ax2, ay2 = a
    bx1 = bs[:, 0]; by1 = bs[:, 1]; bx2 = bs[:, 2]; by2 = bs[:, 3]
    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)
    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)

def _nms(boxes, scores, iou_thres=0.45):
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        ious = _iou(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious < iou_thres]
    return keep

def _undo_letterbox_xyxy(x1, y1, x2, y2, r, dw, dh, W0, H0):
    x1 = (x1 - dw) / r;  y1 = (y1 - dh) / r
    x2 = (x2 - dw) / r;  y2 = (y2 - dh) / r
    x1 = np.clip(x1, 0, W0 - 1); y1 = np.clip(y1, 0, H0 - 1)
    x2 = np.clip(x2, 0, W0 - 1); y2 = np.clip(y2, 0, H0 - 1)
    return x1, y1, x2, y2

# --------- Auto-detecting postprocess ---------
def postprocess(det_out, orig_shape, r, dw, dh, conf_thres=0.25, iou_thres=0.45, names=COCO_CLASSES):
    """
    Robust YOLO-like decoder. Accepts:
      - Output shaped (N, P, 5+nc) or (P, 5+nc) or (5+nc, P)
      - Coords either cxcywh or xyxy
      - Coords either normalized [0,1] or absolute [0, input_size]
      - Obj/cls either logits or probs (auto-sigmoid if needed)

    Returns list of dicts:
      {"bbox":[x,y,w,h], "conf":float, "class_id":int, "label":str}
      bbox is in ORIGINAL image pixels (after unletterbox).
    """
    H0, W0 = orig_shape[:2]
    out = np.array(det_out[0])

    # ---- reshape to (P, 5+nc)
    if out.ndim == 3:    # (N, P, C)
        out = out[0]
    if out.shape[0] < out.shape[1] and out.shape[0] <= 10:
        # (5+nc, P) -> (P, 5+nc)
        out = out.T
    if out.shape[1] < 6:
        return []

    coords = out[:, :4].astype(np.float32)
    obj = out[:, 4].astype(np.float32)
    cls = out[:, 5:].astype(np.float32)

    # ---- auto-sigmoid if values look like logits
    def needs_sigmoid(arr):
        return (np.mean(arr < 0) > 0.01) or (np.mean(arr > 1.5) > 0.01)

    if needs_sigmoid(obj):
        obj = _sigmoid(obj)
    if needs_sigmoid(cls):
        cls = _sigmoid(cls)

    cls_id = cls.argmax(axis=1)
    cls_score = cls.max(axis=1)
    conf = obj * cls_score

    keep = conf >= conf_thres
    if not np.any(keep):
        return []
    coords = coords[keep]; conf = conf[keep]; cls_id = cls_id[keep]

    # ---- detect coord format & scale
    # Heuristic: if many rows satisfy x2>x1 & y2>y1, likely xyxy
    xyxy_like = np.mean((coords[:, 2] > coords[:, 0]) & (coords[:, 3] > coords[:, 1])) > 0.5

    if xyxy_like:
        x1, y1, x2, y2 = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
        maxv = float(np.max(coords)) if coords.size else 0.0
        # If normalized, scale to input size 640 (assumed)
        if maxv <= 1.5:
            x1 *= 640; y1 *= 640; x2 *= 640; y2 *= 640
    else:
        # assume cxcywh
        cx, cy, w, h = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
        maxv = float(np.max(coords)) if coords.size else 0.0
        if maxv <= 1.5:  # normalized -> scale
            cx *= 640; cy *= 640; w *= 640; h *= 640
        x1 = cx - w / 2; y1 = cy - h / 2
        x2 = cx + w / 2; y2 = cy + h / 2

    # ---- undo letterbox to original image
    x1, y1, x2, y2 = _undo_letterbox_xyxy(x1, y1, x2, y2, r, dw, dh, W0, H0)
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    # ---- NMS
    if boxes_xyxy.shape[0] > 1:
        keep_idx = _nms(boxes_xyxy, conf, iou_thres=iou_thres)
        boxes_xyxy = boxes_xyxy[keep_idx]
        conf = conf[keep_idx]
        cls_id = cls_id[keep_idx]

    # ---- package
    dets = []
    for (xa, ya, xb, yb), c, k in zip(boxes_xyxy, conf, cls_id):
        w = max(0.0, xb - xa); h = max(0.0, yb - ya)
        if w < 2 or h < 2:
            continue
        lbl = names[int(k)] if (names and 0 <= int(k) < len(names)) else str(int(k))
        dets.append({
            "bbox": [int(xa), int(ya), int(w), int(h)],
            "conf": float(c),
            "class_id": int(k),
            "label": lbl
        })
    return dets

# --------- Color naming ---------
def bbox_color_name(bgr, bbox):
    """
    Returns a coarse color name from the ROI average in HSV.
    """
    x, y, w, h = map(int, bbox)
    H, W = bgr.shape[:2]
    if w <= 1 or h <= 1:
        return None
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    roi = bgr[y:y + h, x:x + w]
    if roi.size == 0:
        return None

    # Average HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mean_h, mean_s, mean_v = hsv.reshape(-1, 3).mean(axis=0)

    # Simple buckets
    if mean_v < 50:
        return "black"
    if mean_s < 40 and mean_v > 200:
        return "white"
    if mean_s < 40:
        return "gray"
    if mean_h < 10 or mean_h >= 170:
        return "red"
    if 10 <= mean_h < 25:
        return "orange"
    if 25 <= mean_h < 35:
        return "yellow"
    if 35 <= mean_h < 85:
        return "green"
    if 85 <= mean_h < 130:
        return "blue"
    if 130 <= mean_h < 160:
        return "purple"
    return "pink"
