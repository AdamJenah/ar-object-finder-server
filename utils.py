import cv2
import numpy as np

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

def preprocess_bgr_to_yolo_input(bgr, size=640):
    """Letterbox to square, NCHW float32 [0,1], and return scaling params."""
    h0, w0 = bgr.shape[:2]
    r = min(size / h0, size / w0)
    new_w, new_h = int(round(w0 * r)), int(round(h0 * r))
    resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    dw, dh = (size - new_w) // 2, (size - new_h) // 2
    canvas[dh:dh+new_h, dw:dw+new_w] = resized

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    x = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]  # (1,3,H,W)
    return x, r, dw, dh

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

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

def _iou(a, bs):
    ax1, ay1, ax2, ay2 = a
    bx1 = bs[:,0]; by1 = bs[:,1]; bx2 = bs[:,2]; by2 = bs[:,3]
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

def postprocess(outputs, orig_shape, r, dw, dh, conf_thres=0.25, iou_thres=0.45, names=COCO_CLASSES):
    """
    Decode YOLO ONNX output into pixel-space xywh on the original image.
    Expected ONNX output: (N, num_preds, 5+num_classes) with [cx,cy,w,h,obj,cls...].
    If your model returns (num_preds, N, …) or channels-first, transpose accordingly.
    """
    H0, W0 = orig_shape[:2]

    out = outputs[0]
    # Handle common alternative layouts:
    if out.ndim == 3:  # (N, num_preds, 5+nc)
        preds = out[0]
    elif out.ndim == 2:  # (num_preds, 5+nc)
        preds = out
    else:
        raise ValueError(f"Unexpected ONNX output shape: {out.shape}")

    # if your export is (5+nc, num_preds) -> transpose:
    if preds.shape[0] < preds.shape[1] and preds.shape[1] > 10 and preds.shape[0] <= 10:
        preds = preds.T  # make it (num_preds, 5+nc)

    num_cols = preds.shape[1]
    if num_cols < 6:
        raise ValueError(f"Not enough columns in output: {preds.shape}")

    xywh = preds[:, 0:4]
    obj  = preds[:, 4]
    cls  = preds[:, 5:]

    # convert logits to probabilities when needed
    # (most YOLO ONNX exports already output sigmoid; but your raw values suggest logits)
    obj = _sigmoid(obj)
    cls = _sigmoid(cls)

    cls_id = cls.argmax(axis=1)
    cls_score = cls.max(axis=1)
    conf = obj * cls_score

    # threshold
    keep = conf >= conf_thres
    xywh = xywh[keep]
    conf = conf[keep]
    cls_id = cls_id[keep]

    if xywh.shape[0] == 0:
        return []

    # cx,cy,w,h -> x1,y1,x2,y2 in letterboxed space
    cx, cy, w, h = xywh[:,0], xywh[:,1], xywh[:,2], xywh[:,3]
    x1 = cx - w/2
    y1 = cy - h/2
    x2 = cx + w/2
    y2 = cy + h/2

    # undo letterbox to original image space
    x1 = (x1 - dw) / r
    y1 = (y1 - dh) / r
    x2 = (x2 - dw) / r
    y2 = (y2 - dh) / r

    # clip to image
    x1 = np.clip(x1, 0, W0 - 1)
    y1 = np.clip(y1, 0, H0 - 1)
    x2 = np.clip(x2, 0, W0 - 1)
    y2 = np.clip(y2, 0, H0 - 1)

    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    # NMS
    if boxes_xyxy.shape[0] > 1:
        keep_idx = _nms(boxes_xyxy, conf, iou_thres=iou_thres)
        boxes_xyxy = boxes_xyxy[keep_idx]
        conf = conf[keep_idx]
        cls_id = cls_id[keep_idx]

    # package results as xywh pixels
    results = []
    for (x1, y1, x2, y2), c, k in zip(boxes_xyxy, conf, cls_id):
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        if w < 2 or h < 2:
            continue  # skip degenerate boxes
        label = names[k] if 0 <= k < len(names) else str(int(k))
        results.append({
            "bbox": [int(x1), int(y1), int(w), int(h)],
            "conf": float(c),
            "class_id": int(k),
            "label": label
        })
    return results

def bbox_color_name(bgr, bbox):
    x, y, w, h = map(int, bbox)
    H, W = bgr.shape[:2]
    if w <= 1 or h <= 1:  # guard against empty ROI
        return None
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    roi = bgr[y:y+h, x:x+w]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mean_hsv = hsv.reshape(-1, 3).mean(axis=0)
    h_, s, v = mean_hsv
    if v < 50: return "black"
    if s < 40 and v > 200: return "white"
    if s < 40: return "gray"
    if h_ < 10 or h_ >= 170: return "red"
    if 10 <= h_ < 25: return "orange"
    if 25 <= h_ < 35: return "yellow"
    if 35 <= h_ < 85: return "green"
    if 85 <= h_ < 130: return "blue"
    if 130 <= h_ < 160: return "purple"
    return "pink"
