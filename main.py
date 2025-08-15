import os, cv2, numpy as np, onnxruntime as ort
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from utils import preprocess_bgr_to_yolo_input, postprocess, bbox_color_name
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

    # pick threshold (fallback to env)
    th = CONF_THRES if conf_thresh is None else float(conf_thresh)

    detections = postprocess(
        det_out, bgr.shape, r, dw, dh,
        conf_thres=th,
        input_size=INPUT_SIZE,
        names=None  # or pass your class list if you have one
    )



    response = {"mode":"objects", "match": None, "detections": detections}

    # Optional: pose branch
    if (use_pose == "true") and (target_clothing in {"shirt","pants"}):
        people = run_pose(pose_session, bgr)  # TODO: implement parsing in pose_utils.run_pose
        rois = clothing_rois_from_keypoints(people, target=target_clothing)
        clothing_hits = []
        for roi in rois:
            col = bbox_color_name(bgr, roi)
            clothing_hits.append({"target": target_clothing, "color": col, "bbox": roi})
        response["mode"] = "pose"
        response["pose"] = {"clothing": clothing_hits}
        if color:
            response["match"] = any(hit["color"] == color for hit in clothing_hits)
        else:
            response["match"] = len(clothing_hits) > 0

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
