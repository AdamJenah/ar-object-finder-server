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

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    h, w = im.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2; dh /= 2
    if (w, h) != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh-0.1)), int(round(dh+0.1))
    left, right = int(round(dw-0.1)), int(round(dw+0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)

def preprocess_bgr_to_yolo_input(image_bgr, size=640):
    img, r, (dw, dh) = letterbox(image_bgr, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None, ...]  # NCHW
    return img, r, dw, dh

def postprocess(outputs, orig_shape, r, dw, dh, conf_thres=0.25):
    preds = outputs[0][0]  # (num, 85) -> [cx,cy,w,h,conf,80 scores]
    boxes = preds[:, :4]
    obj_conf = preds[:, 4:5]
    cls_scores = preds[:, 5:]
    cls_ids = np.argmax(cls_scores, axis=1)
    cls_conf = cls_scores[np.arange(cls_scores.shape[0]), cls_ids][:, None]
    conf = obj_conf * cls_conf

    mask = conf[:, 0] >= conf_thres
    boxes, conf, cls_ids = boxes[mask], conf[mask, 0], cls_ids[mask]

    out = []
    oh, ow = orig_shape[:2]
    for b, c, ci in zip(boxes, conf, cls_ids):
        cx, cy, w, h = b
        x1 = cx - w/2 - dw
        y1 = cy - h/2 - dh
        x2 = cx + w/2 - dw
        y2 = cy + h/2 - dh
        x1, y1, x2, y2 = x1 / r, y1 / r, x2 / r, y2 / r
        x1 = max(0, min(int(x1), ow-1))
        y1 = max(0, min(int(y1), oh-1))
        x2 = max(0, min(int(x2), ow-1))
        y2 = max(0, min(int(y2), oh-1))
        out.append({
            "bbox":[int(x1), int(y1), int(x2), int(y2)],
            "confidence": float(c),
            "class_id": int(ci),
            "label": COCO_CLASSES[int(ci)] if 0 <= ci < len(COCO_CLASSES) else str(int(ci))
        })
    return out

def bbox_color_name(image_bgr, bbox):
    x1, y1, x2, y2 = bbox
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mean_hsv = hsv.reshape(-1, 3).mean(axis=0)
    h, s, v = mean_hsv
    if v < 50: return "black"
    if s < 40 and v > 200: return "white"
    if s < 40: return "gray"
    if h < 10 or h >= 170: return "red"
    if 10 <= h < 25: return "orange"
    if 25 <= h < 35: return "yellow"
    if 35 <= h < 85: return "green"
    if 85 <= h < 130: return "blue"
    if 130 <= h < 160: return "purple"
    return "unknown"
