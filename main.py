import os, cv2, numpy as np, onnxruntime as ort
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from utils import preprocess_bgr_to_yolo_input, postprocess_onnx_nms, bbox_color_name, COCO_CLASSES
from pose_utils import run_pose, clothing_rois_from_keypoints

OBJ_MODEL_PATH  = os.getenv("OBJ_MODEL_PATH",  "models/yolov12n.onnx")
POSE_MODEL_PATH = os.getenv("POSE_MODEL_PATH", "models/yolov11n-pose.onnx")
INPUT_SIZE = int(os.getenv("INPUT_SIZE", "640"))
CONF_THRES = float(os.getenv("CONF_THRES", "0.25"))

app = FastAPI(title="Objects + Pose Inference API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

obj_session  = ort.InferenceSession(OBJ_MODEL_PATH,  providers=["CPUExecutionProvider"])
pose_session = ort.InferenceSession(POSE_MODEL_PATH, providers=["CPUExecutionProvider"])

@app.get("/")
def health():
    return {"status":"ok"}

@app.post("/infer")
async def infer(
    image: UploadFile = File(...),
    object: str | None = Form(default=None),
    color:  str | None = Form(default=None),
    use_pose: str | None = Form(default="false"),
    target_clothing: str | None = Form(default=None),
    conf_thresh: float | None = Form(default=None),   # ← renamed & typed
):
    img_bytes = await image.read()
    bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return {"error":"invalid_image"}

    # objects (always)
    inp, r, dw, dh = preprocess_bgr_to_yolo_input(bgr, size=INPUT_SIZE)
    det_out = obj_session.run(None, {obj_session.get_inputs()[0].name: inp})

    # If you added a client slider 'conf_thresh', use it; else fall back to CONF_THRES
    score_thres = conf_thresh if conf_thresh is not None else CONF_THRES

    detections = postprocess_onnx_nms(det_out, bgr.shape, r, dw, dh, score_thres=score_thres, names=COCO_CLASSES)  # pass your class list if custom

    # add ROI color
    for d in detections:
        d["color"] = bbox_color_name(bgr, d["bbox"])

    response = {"mode":"objects", "match": None, "detections": detections}

    def norm(s): 
        return (s or "").strip().lower()

    obj_q = norm(object)
    col_q = norm(color)

    if response["mode"] == "objects" and (obj_q or col_q):
        response["match"] = False
        for d in detections:
            lbl = norm(d.get("label"))
            col = norm(d.get("color"))
            obj_ok = (not obj_q) or (obj_q in lbl)  # substring
            col_ok = (not col_q) or (col_q == col)  # strict
            if obj_ok and col_ok:
                response["match"] = True
                response["matched_detection"] = d
                break

    # Optional: pose branch
    pose_on = str(use_pose or "").lower() == "true"
    tgt = (target_clothing or "").strip().lower()
    if pose_on:
        if tgt not in {"shirt", "pants"}:
            tgt = "shirt"   # sensible default so pose still runs
        people = run_pose(pose_session, bgr)  # make sure this returns keypoints!
        rois = clothing_rois_from_keypoints(people, target=tgt)
        clothing_hits = []
        for roi in rois:
            col = bbox_color_name(bgr, roi)
            clothing_hits.append({"target": tgt, "color": col, "bbox": roi})
        response["mode"] = "pose"
        response["pose"] = {"clothing": clothing_hits}
        # pose match: color optional
        response["match"] = (any(norm(h["color"]) == col_q for h in clothing_hits) 
                            if col_q else len(clothing_hits) > 0)

    # Object+color prompt match (when not using pose)
    if response["mode"] == "objects" and (object or color):
        found = False
        for d in detections:
            obj_ok = (object is None) or (d["label"] == object)
            col_ok = (color  is None) or (d["color"] == color)
            if obj_ok and col_ok:
                response["match"] = True
                response["matched_detection"] = d
                found = True
                break
        if not found:
            response["match"] = False

    return response
