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
    inter = np.maximum(0.0, inter_x2 - inter_x1) * np.maximum(0.0, inter_y2 - inter_y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)

def _nms(boxes, scores, iou_thres=0.45):
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1: break
        ious = _iou(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious < iou_thres]
    return keep

def _undo_letterbox_xyxy(x1, y1, x2, y2, r, dw, dh, W0, H0):
    x1 = (x1 - dw) / r;  y1 = (y1 - dh) / r
    x2 = (x2 - dw) / r;  y2 = (y2 - dh) / r
    x1 = np.clip(x1, 0, W0 - 1); y1 = np.clip(y1, 0, H0 - 1)
    x2 = np.clip(x2, 0, W0 - 1); y2 = np.clip(y2, 0, H0 - 1)
    return x1, y1, x2, y2

def _needs_sigmoid(arr):
    return (np.mean(arr < 0) > 0.01) or (np.mean(arr > 1.5) > 0.01)

def _coords_to_xyxy(coords, input_size):
    """Auto-handle xyxy vs cxcywh, normalized vs absolute."""
    coords = coords.astype(np.float32)
    # Heuristic: if many rows satisfy x2>x1 & y2>y1, treat as xyxy
    xyxy_like = np.mean((coords[:,2] > coords[:,0]) & (coords[:,3] > coords[:,1])) > 0.5
    if xyxy_like:
        x1, y1, x2, y2 = coords[:,0], coords[:,1], coords[:,2], coords[:,3]
        if float(np.max(coords)) <= 1.5:
            x1 *= input_size; y1 *= input_size; x2 *= input_size; y2 *= input_size
    else:
        cx, cy, w, h = coords[:,0], coords[:,1], coords[:,2], coords[:,3]
        if float(np.max(coords)) <= 1.5:
            cx *= input_size; cy *= input_size; w *= input_size; h *= input_size
        x1 = cx - w/2; y1 = cy - h/2; x2 = cx + w/2; y2 = cy + h/2
    return x1, y1, x2, y2

def _reshape_pred(a):
    """Return (P, C) for any (N,P,C), (P,C), or (C,P)."""
    a = np.array(a)
    if a.ndim == 3:
        a = a[0]
    if a.shape[0] < a.shape[1] and a.shape[0] <= 10:
        a = a.T
    return a

def _decode_fused(fused, conf_thres, input_size):
    """Fused layout: (P, 4+K) or (P, 5+K) with/without obj."""
    fused = _reshape_pred(fused)
    if fused.shape[1] < 6:
        return None
    coords = fused[:, :4]
    rest   = fused[:, 4:]

    # try with obj and without obj; pick better
    cands = []

    # with objectness
    if rest.shape[1] >= 2:
        obj = rest[:, 0]
        cls = rest[:, 1:]
        if _needs_sigmoid(obj): obj = _sigmoid(obj)
        if _needs_sigmoid(cls): cls = _sigmoid(cls)
        cls_id = cls.argmax(axis=1); cls_score = cls.max(axis=1)
        conf = obj * cls_score
        keep = conf >= conf_thres
        if np.any(keep):
            x1,y1,x2,y2 = _coords_to_xyxy(coords[keep], input_size)
            w = x2 - x1; h = y2 - y1
            ok = (w >= 2) & (h >= 2)
            if np.any(ok):
                cands.append(("fused+obj", x1[ok],y1[ok],x2[ok],y2[ok], conf[keep][ok], cls_id[keep][ok]))

    # without objectness
    cls = rest
    if _needs_sigmoid(cls): cls = _sigmoid(cls)
    cls_id = cls.argmax(axis=1); conf = cls.max(axis=1)
    keep = conf >= conf_thres
    if np.any(keep):
        x1,y1,x2,y2 = _coords_to_xyxy(coords[keep], input_size)
        w = x2 - x1; h = y2 - y1
        ok = (w >= 2) & (h >= 2)
        if np.any(ok):
            cands.append(("fused-noobj", x1[ok],y1[ok],x2[ok],y2[ok], conf[keep][ok], cls_id[keep][ok]))

    return cands or None

def _decode_decoupled(boxes, scores, conf_thres, input_size):
    """Two-output layout: boxes (P,4) & scores (P,K), arbitrary transposes."""
    boxes = np.array(boxes); scores = np.array(scores)

    # move to (P,4) and (P,K)
    if boxes.ndim == 3: boxes = boxes[0]
    if scores.ndim == 3: scores = scores[0]
    if boxes.shape[0] == 4 and boxes.shape[1] != 4: boxes = boxes.T
    if scores.shape[0] < scores.shape[1] and scores.shape[0] <= 10: scores = scores.T

    if boxes.shape[1] != 4 or scores.ndim != 2: 
        return None

    if _needs_sigmoid(scores): scores = _sigmoid(scores)
    cls_id = scores.argmax(axis=1); conf = scores.max(axis=1)
    keep = conf >= conf_thres
    if not np.any(keep): 
        return None

    x1,y1,x2,y2 = _coords_to_xyxy(boxes[keep], input_size)
    w = x2 - x1; h = y2 - y1
    ok = (w >= 2) & (h >= 2)
    if not np.any(ok): 
        return None

    return [("decoupled", x1[ok],y1[ok],x2[ok],y2[ok], conf[keep][ok], cls_id[keep][ok])]

def postprocess_onnx_nms(outputs, orig_shape, r, dw, dh, score_thres=0.25, names=None):
    """
    Parse ONNX exported with nms=True.
    outputs[0].shape == (1, num, 6): [x1,y1,x2,y2,score,class_id] in letterboxed pixels.
    Return [{"bbox":[x,y,w,h], "conf":score, "class_id":k, "label": name}, ...] in ORIGINAL image pixels.
    """
    H0, W0 = orig_shape[:2]
    out = np.array(outputs[0])
    if out.ndim != 3 or out.shape[2] < 6:
        return []

    dets = out[0]  # (num, 6)
    if dets.size == 0:
        return []

    x1, y1, x2, y2, score, cid = dets[:,0], dets[:,1], dets[:,2], dets[:,3], dets[:,4], dets[:,5].astype(int)

    # filter by score
    keep = score >= score_thres
    if not np.any(keep):
        return []
    x1, y1, x2, y2, score, cid = x1[keep], y1[keep], x2[keep], y2[keep], score[keep], cid[keep]

    # undo letterbox
    x1 = (x1 - dw) / r;  y1 = (y1 - dh) / r
    x2 = (x2 - dw) / r;  y2 = (y2 - dh) / r

    # clip
    x1 = np.clip(x1, 0, W0-1); y1 = np.clip(y1, 0, H0-1)
    x2 = np.clip(x2, 0, W0-1); y2 = np.clip(y2, 0, H0-1)

    results = []
    for xa, ya, xb, yb, s, k in zip(x1, y1, x2, y2, score, cid):
        w = max(0.0, xb - xa); h = max(0.0, yb - ya)
        if w < 2 or h < 2: 
            continue
        label = (names[k] if names and 0 <= k < len(names) else str(int(k)))
        results.append({
            "bbox": [int(xa), int(ya), int(w), int(h)],
            "conf": float(s),
            "class_id": int(k),
            "label": label
        })
    return results


# --------- Color naming ---------
def bbox_color_name(bgr, bbox, inset_ratio=0.08):
    """
    Robust color naming:
    - Ignores a border around the box (to avoid client-drawn overlays or edges)
    - Uses dominant hue weighted by saturation, not a simple average
    - Handles black/white/gray as special cases
    """
    x, y, w, h = map(int, bbox)
    H, W = bgr.shape[:2]
    if w <= 2 or h <= 2:
        return None

    # Inset ROI to avoid borders/box strokes (e.g., Streamlit's lime outline)
    dx = int(w * inset_ratio * 0.5)
    dy = int(h * inset_ratio * 0.5)
    xi = max(0, min(W - 1, x + dx))
    yi = max(0, min(H - 1, y + dy))
    wi = max(1, min(W - xi, w - 2 * dx))
    hi = max(1, min(H - yi, h - 2 * dy))

    roi = bgr[yi:yi + hi, xi:xi + wi]
    if roi.size == 0:
        return None

    # Light blur to reduce noise
    roi = cv2.GaussianBlur(roi, (3, 3), 0)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    Hc = hsv[..., 0]  # 0..180
    S  = hsv[..., 1]  # 0..255
    V  = hsv[..., 2]  # 0..255

    # Grayscale buckets first
    if np.mean(V) < 45:
        return "black"
    if np.mean(S) < 28:  # low saturation overall
        return "white" if np.mean(V) > 200 else "gray"

    # Mask out very low saturation / value pixels before hue voting
    mask = (S >= 30) & (V >= 40)
    if not np.any(mask):
        return "gray"

    Hm = Hc[mask].astype(np.int32)
    Sw = S[mask].astype(np.float32)

    # Hue histogram weighted by saturation (stronger colors count more)
    # OpenCV hue range is 0..180
    hist = np.bincount(Hm, weights=Sw, minlength=181)
    h_mode = int(hist.argmax())

    # Map hue to color name
    # (bounds tuned for OpenCV HSV; tweak as you like)
    if h_mode < 10 or h_mode >= 170: return "red"
    if h_mode < 25:  return "orange"
    if h_mode < 35:  return "yellow"
    if h_mode < 85:  return "green"
    if h_mode < 125: return "blue"
    if h_mode < 150: return "purple"
    return "pink"