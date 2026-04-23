import json
import os
import sys
from pathlib import Path

import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
if VENV_PY.exists() and Path(sys.executable).resolve() != VENV_PY.resolve():
    os.execv(str(VENV_PY), [str(VENV_PY), *sys.argv])
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config_manager import Config
from services.face_service import FaceRecognitionService


TEMPP_DIR = Path(__file__).resolve().parent
REGISTER_DIR = TEMPP_DIR / "registered_faces"
OUTPUT_DIR = TEMPP_DIR / "outputs"
REPORT_DIR = TEMPP_DIR / "reports"

REGISTER_THRESHOLD = 0.15
RECOGNITION_THRESHOLD = 0.6
DETECTION_THRESHOLDS = [round(v / 100.0, 2) for v in range(50, 0, -1)]
REGISTER_IMGSZ = 5120
INFER_IMGSZ = 5120
SOURCE_IMAGE = ROOT / "test.jpg"


def ensure_dirs():
    for path in (REGISTER_DIR, OUTPUT_DIR, REPORT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def detect_faces_with_imgsz(detector, image, conf_threshold: float, imgsz: int, align: bool = True, output_size: int = 112):
    if detector is None or getattr(detector, "model", None) is None:
        return []

    results = detector.model.predict(
        image,
        conf=float(conf_threshold),
        device=getattr(detector, "device", None),
        imgsz=int(imgsz),
        verbose=False,
    )
    if not results:
        return []

    result = results[0]
    if result.boxes is None or len(result.boxes) <= 0:
        return []

    detections = []
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()

    for i, box in enumerate(boxes):
        det_info = {
            "box": box.astype(int).tolist(),
            "score": float(confs[i]),
            "keypoints": None,
            "aligned_face": None,
        }
        if align and result.keypoints is not None and i < len(result.keypoints.xy):
            try:
                kpts = result.keypoints.xy[i].cpu().numpy()
                det_info["keypoints"] = kpts.tolist()
                det_info["aligned_face"] = detector.align_face(image, kpts, output_size)
            except Exception:
                pass

        if det_info["aligned_face"] is None:
            x1, y1, x2, y2 = map(int, box)
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            face = image[y1:y2, x1:x2]
            if face.size > 0:
                det_info["aligned_face"] = cv2.resize(face, (output_size, output_size))

        if det_info["aligned_face"] is not None:
            detections.append(det_info)

    return detections


def sort_detections_reading_order(detections, row_tolerance: int = 85):
    if not detections:
        return []

    def center_xy(det):
        x1, y1, x2, y2 = det["box"]
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    rows = []
    for det in sorted(detections, key=lambda item: center_xy(item)[1]):
        _, cy = center_xy(det)
        target_row = None
        for row in rows:
            if abs(row["cy"] - cy) <= row_tolerance:
                target_row = row
                break
        if target_row is None:
            rows.append({"cy": cy, "items": [det]})
        else:
            target_row["items"].append(det)
            count = len(target_row["items"])
            target_row["cy"] = ((target_row["cy"] * (count - 1)) + cy) / count

    ordered = []
    for row in sorted(rows, key=lambda item: item["cy"]):
        ordered.extend(sorted(row["items"], key=lambda item: center_xy(item)[0]))
    return ordered


def make_label(index: int):
    return f"test_{index}"


def register_gallery(service: FaceRecognitionService, image):
    detections = detect_faces_with_imgsz(
        detector=service.detector,
        image=image,
        conf_threshold=REGISTER_THRESHOLD,
        imgsz=REGISTER_IMGSZ,
        align=True,
        output_size=112,
    )
    detections = sort_detections_reading_order(detections)

    service.known_faces.clear()
    service.student_labels.clear()

    gallery = []
    for idx, det in enumerate(detections, start=1):
        face_bgr = det.get("aligned_face")
        if face_bgr is None:
            continue

        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        feature = service.extract_feature(Image.fromarray(face_rgb))
        if feature is None:
            continue

        name = make_label(idx)
        face_path = REGISTER_DIR / f"{name}.jpg"
        cv2.imwrite(str(face_path), face_bgr)

        service.upsert_known_face(idx, name, feature, None)
        gallery.append(
            {
                "id": idx,
                "name": name,
                "score": round(float(det["score"]), 4),
                "box": [int(v) for v in det["box"]],
                "face_path": str(face_path.relative_to(TEMPP_DIR)),
            }
        )

    return gallery


def recognize_faces(service: FaceRecognitionService, image, conf_threshold: float):
    detections = detect_faces_with_imgsz(
        detector=service.detector,
        image=image,
        conf_threshold=conf_threshold,
        imgsz=INFER_IMGSZ,
        align=True,
        output_size=112,
    )

    recognized = []
    for det in detections:
        face_bgr = det.get("aligned_face")
        if face_bgr is None:
            continue

        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        feature = service.extract_feature(Image.fromarray(face_rgb))
        if feature is None:
            continue

        best_name = None
        best_score = 0.0
        for student_id, known_feat in service.known_faces.items():
            is_same, score = service.is_same_person(feature, known_feat, threshold=RECOGNITION_THRESHOLD)
            if score > best_score:
                best_score = float(score)
                if is_same:
                    best_name = service.student_labels.get(student_id)

        if best_name:
            recognized.append(
                {
                    "box": [int(v) for v in det["box"]],
                    "name": best_name,
                    "score": round(best_score, 4),
                }
            )

    return recognized, len(detections)


def draw_recognized_only(image, results):
    output = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for result in results:
        x1, y1, x2, y2 = result["box"]
        name = result["name"]

        cv2.rectangle(output, (x1, y1), (x2, y2), (25, 135, 84), 2, lineType=cv2.LINE_AA)
        text_y = max(20, y1 - 8)
        cv2.putText(output, name, (x1, text_y), font, 0.68, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(output, name, (x1, text_y), font, 0.68, (25, 135, 84), 2, cv2.LINE_AA)

    return output


def main():
    ensure_dirs()

    if not SOURCE_IMAGE.exists():
        raise FileNotFoundError(f"找不到图片: {SOURCE_IMAGE}")

    image = cv2.imread(str(SOURCE_IMAGE))
    if image is None:
        raise RuntimeError(f"无法读取图片: {SOURCE_IMAGE}")

    cfg = Config("config/config.yaml")
    cfg.recognition["similarity_threshold"] = RECOGNITION_THRESHOLD

    service = FaceRecognitionService(config=cfg)
    service.similarity_threshold = RECOGNITION_THRESHOLD

    gallery = register_gallery(service, image)
    if not gallery:
        raise RuntimeError("自动注册失败，未获得可用人脸样本。")

    summary = {
        "source_image": str(SOURCE_IMAGE.relative_to(ROOT)),
        "register_threshold": REGISTER_THRESHOLD,
        "recognition_threshold": RECOGNITION_THRESHOLD,
        "register_imgsz": REGISTER_IMGSZ,
        "infer_imgsz": INFER_IMGSZ,
        "registered_count": len(gallery),
        "registered_faces": gallery,
        "runs": [],
    }

    for threshold in DETECTION_THRESHOLDS:
        recognized, detected_count = recognize_faces(service, image, threshold)
        drawn = draw_recognized_only(image, recognized)
        output_name = f"test_conf_{threshold:.2f}.jpg"
        output_path = OUTPUT_DIR / output_name
        cv2.imwrite(str(output_path), drawn)

        summary["runs"].append(
            {
                "detection_threshold": round(threshold, 2),
                "detected_count": detected_count,
                "recognized_count": len(recognized),
                "recognized_names": [item["name"] for item in recognized],
                "output_file": str(output_path.relative_to(TEMPP_DIR)),
            }
        )
        print(
            f"RUN_OK threshold={threshold:.2f} detected={detected_count} "
            f"recognized={len(recognized)} output={output_path.name}"
        )

    summary_path = REPORT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"EXPERIMENT_DONE registered={len(gallery)} thresholds={len(DETECTION_THRESHOLDS)} "
        f"summary={summary_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
