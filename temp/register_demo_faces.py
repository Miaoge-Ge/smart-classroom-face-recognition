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
from core.crypto_manager import encrypt_bytes, encrypt_to_b64
from core.database import SessionLocal
from core.models import Student
from services.face_service import FaceRecognitionService


def _pick_demo_image() -> Path:
    p = ROOT / "demo.png"
    if p.exists():
        return p
    p = ROOT / "demo.jpg"
    if p.exists():
        return p
    p = Path(__file__).resolve().parent / "demo.png"
    if p.exists():
        return p
    p = Path(__file__).resolve().parent / "demo.jpg"
    if p.exists():
        return p
    raise FileNotFoundError("找不到 demo.png（也未找到 demo.jpg）")


def _detect_faces_with_imgsz(detector, image, imgsz: int, align: bool = True, output_size: int = 112):
    if detector is None or getattr(detector, "model", None) is None:
        return []
    results = detector.model.predict(
        image,
        conf=float(getattr(detector, "conf_threshold", 0.05)),
        device=getattr(detector, "device", None),
        imgsz=int(imgsz),
        verbose=False,
    )
    if not results:
        return []
    result = results[0]
    detections = []
    if result.boxes is None or len(result.boxes) <= 0:
        return []
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


def main():
    img_path = _pick_demo_image()
    image = cv2.imread(str(img_path))
    if image is None:
        raise RuntimeError(f"无法读取图片: {img_path}")

    cfg = Config()
    det_cfg = cfg.detector or {}
    model_path = det_cfg.get("model_path")
    if not model_path:
        raise RuntimeError("detector.model_path 未配置")
    base_conf = float(det_cfg.get("conf_threshold", 0.05))
    det_conf = float(os.environ.get("DET_CONF_THRESHOLD", str(min(base_conf, 0.05))))
    cfg.detector["conf_threshold"] = det_conf

    service = FaceRecognitionService(config=cfg)
    if not service.detector or not hasattr(service.detector, "detect_faces"):
        raise RuntimeError("检测模型未初始化或不支持 detect_faces")

    imgsz = int(os.environ.get("DET_IMGSZ", "2560"))
    detections = _detect_faces_with_imgsz(service.detector, image, imgsz=imgsz, align=True, output_size=112)
    detections = [d for d in detections if d.get("aligned_face") is not None]
    detections.sort(key=lambda d: float(d.get("score") or 0.0), reverse=True)
    if not detections:
        raise RuntimeError("demo 图中未检测到人脸")

    ok = 0
    db = SessionLocal()
    try:
        faces_dir = ROOT / "data" / "faces"
        faces_dir.mkdir(parents=True, exist_ok=True)

        for i, det in enumerate(detections, start=1):
            face = det.get("aligned_face")
            if face is None:
                continue
            name = f"test_{i}"
            student_no = name

            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(face_rgb)
            feat = service.extract_feature(pil_img)
            if feat is None:
                print(f"REGISTER_SKIP {name} reason=extract_feature_failed")
                continue

            embedding_raw = feat.detach().cpu().numpy().astype("float32").tobytes()
            embedding_enc = encrypt_to_b64(embedding_raw)

            stu = db.query(Student).filter(Student.student_no == student_no).first()
            if not stu:
                stu = Student(
                    name=name,
                    student_no=student_no,
                    college="test",
                    gender="男",
                    class_name="test",
                )
                db.add(stu)
                db.flush()
            else:
                stu.name = name
                stu.college = "test"
                stu.gender = "男"
                stu.class_name = "test"

            ok2, buf = cv2.imencode(".png", face)
            if not ok2:
                print(f"REGISTER_SKIP {name} reason=encode_failed")
                continue

            file_path = faces_dir / f"{stu.student_id}.enc"
            file_path.write_bytes(encrypt_bytes(buf.tobytes()))

            stu.face_image_path = str(file_path)
            stu.face_embedding_enc = embedding_enc
            if getattr(service, "model_sig", None):
                stu.face_embedding_model_sig = service.model_sig

            service.upsert_known_face(stu.student_id, stu.name, feat, stu.student_no)
            db.commit()
            ok += 1
            print(f"REGISTER_OK {name}")

    finally:
        db.close()

    print(f"REGISTER_DONE total={len(detections)} ok={ok} det_conf={det_conf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
