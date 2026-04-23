import base64
import os
import sys
import threading
import time
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
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


def _pick_demo_image() -> Path:
    p = ROOT / "demo.jpg"
    if p.exists():
        return p
    p = ROOT / "demo.png"
    if p.exists():
        return p
    p = Path(__file__).resolve().parent / "demo.jpg"
    if p.exists():
        return p
    p = Path(__file__).resolve().parent / "demo.png"
    if p.exists():
        return p
    raise FileNotFoundError("找不到 demo.jpg（也未找到 demo.png）")


def _draw_results(img, results):
    out = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    text_thickness = 2
    line = cv2.LINE_AA

    def draw_text(x: int, y: int, text: str, color):
        cv2.putText(out, text, (x, y), font, scale, (0, 0, 0), text_thickness + 2, lineType=line)
        cv2.putText(out, text, (x, y), font, scale, color, text_thickness, lineType=line)

    for r in results:
        box = r.get("box") or []
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        name = str(r.get("name") or "Unknown")
        tag = r.get("tag")
        is_unknown = name == "Unknown"
        if is_unknown:
            continue
        box_color = (25, 135, 84)
        name_color = (25, 135, 84)
        tag_color = (255, 255, 255)

        h, w = out.shape[:2]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))

        tag_text = None
        if isinstance(tag, str) and tag.strip():
            tag_text = f"[{tag.strip()}]"

        cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 2, lineType=line)

        y_text = max(18, y1 - 6)
        if not tag_text:
            (tw, th), bl = cv2.getTextSize(name, font, scale, text_thickness)
            tx = int((x1 + x2 - tw) / 2)
            tx = max(0, min(w - tw - 1, tx))
            draw_text(tx, y_text, name, name_color)
        else:
            (tw1, th1), bl1 = cv2.getTextSize(name, font, scale, text_thickness)
            (tw2, th2), bl2 = cv2.getTextSize(tag_text, font, scale, text_thickness)
            total_w = tw1 + 6 + tw2
            tx0 = int((x1 + x2 - total_w) / 2)
            tx0 = max(0, min(w - total_w - 1, tx0))
            draw_text(tx0, y_text, name, name_color)
            draw_text(tx0 + tw1 + 6, y_text, tag_text, tag_color)

    return out


def _b64_file(path: Path) -> str:
    data = path.read_bytes()
    return base64.b64encode(data).decode("ascii")

def _detect_faces_with_imgsz(detector, image, imgsz: int, align: bool = True, output_size: int = 112):
    if detector is None or getattr(detector, "model", None) is None:
        return []
    results = detector.model.predict(
        image,
        conf=float(getattr(detector, "conf_threshold", 0.01)),
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


def _recognize_with_imgsz(service: FaceRecognitionService, frame, imgsz: int):
    detections = _detect_faces_with_imgsz(service.detector, frame, imgsz=imgsz, align=True, output_size=112)
    results = []
    for det in detections:
        box = det.get("box")
        aligned_face = det.get("aligned_face")
        if aligned_face is None or not box:
            continue
        face_img_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(face_img_rgb)
        feat = service.extract_feature(pil_img)
        best_name = "Unknown"
        best_tag = None
        best_student_id = None
        best_score = 0.0
        if feat is not None and service.known_faces:
            for student_id, known_feat in service.known_faces.items():
                is_same, score = service.is_same_person(feat, known_feat)
                if score > best_score:
                    best_score = score
                    if is_same:
                        best_student_id = int(student_id)
                        full_label = service.student_labels.get(best_student_id, "Unknown")
                        if isinstance(full_label, str):
                            s = full_label.strip()
                            if "[" in s and s.endswith("]"):
                                best_name = s.split("[", 1)[0].strip() or "Unknown"
                                best_tag = s[s.rfind("[") + 1 : -1].strip() or None
                            else:
                                best_name = s or "Unknown"
        if best_score < service.similarity_threshold:
            best_name = "Unknown"
            best_tag = None
            best_student_id = None
        results.append(
            {
                "box": box,
                "name": best_name,
                "tag": best_tag,
                "student_id": best_student_id,
                "score": float(best_score),
            }
        )
    return results


def main():
    temp_dir = Path(__file__).resolve().parent
    temp_dir.mkdir(parents=True, exist_ok=True)

    img_path = _pick_demo_image()
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"无法读取图片: {img_path}")

    cfg = Config()
    det_cfg = cfg.detector or {}
    base_conf = float(det_cfg.get("conf_threshold", 0.01))
    cfg.detector["conf_threshold"] = float(os.environ.get("DET_CONF_THRESHOLD", str(min(base_conf, 0.01))))
    service = FaceRecognitionService(config=cfg)

    imgsz = int(os.environ.get("DET_IMGSZ", "1280"))
    t0 = time.time()
    results = _recognize_with_imgsz(service, img, imgsz=imgsz)
    ms = (time.time() - t0) * 1000.0

    out = _draw_results(img, results)

    input_path = temp_dir / "demo_input.jpg"
    out_path = temp_dir / "demo_result.jpg"
    html_path = temp_dir / "demo_result.html"

    cv2.imwrite(str(input_path), img)
    cv2.imwrite(str(out_path), out)

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Demo 图片识别</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, 'PingFang SC', 'Noto Sans CJK SC', 'Microsoft YaHei', sans-serif; margin: 0; background: #0b1220; color: #e5e7eb; }}
    .wrap {{ max-width: 1100px; margin: 0 auto; padding: 24px; }}
    .card {{ background: #111827; border: 1px solid #1f2937; border-radius: 12px; padding: 16px; }}
    .row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    h1 {{ font-size: 18px; margin: 0 0 12px; }}
    .meta {{ font-size: 13px; color: #9ca3af; margin: 8px 0 16px; }}
    img {{ width: 100%; height: auto; border-radius: 10px; border: 1px solid #374151; background: #0b1220; }}
    pre {{ background: #0b1220; border: 1px solid #1f2937; padding: 12px; border-radius: 10px; overflow: auto; }}
    @media (max-width: 900px) {{ .row {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>非摄像头模式：demo 图片识别效果</h1>
      <div class="meta">输入：{img_path.name} ｜ imgsz：{imgsz} ｜ 推理耗时：{ms:.1f}ms ｜ 人脸数：{len(results)}</div>
      <div class="row">
        <div>
          <div class="meta">原图（对应“摄像头区域”）</div>
          <img src="demo_input.jpg" alt="input" />
        </div>
        <div>
          <div class="meta">识别后效果（绿：识别姓名｜黑：注册标签）</div>
          <img src="demo_result.jpg" alt="result" />
        </div>
      </div>
      <div class="meta">识别结果 JSON</div>
      <pre>{results}</pre>
    </div>
  </div>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")

    port = int(os.environ.get("DEMO_PORT", "8010"))

    handler = lambda *args, **kwargs: SimpleHTTPRequestHandler(*args, directory=str(temp_dir), **kwargs)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)

    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()

    url = f"http://127.0.0.1:{port}/{html_path.name}"
    webbrowser.open(url)
    print(f"OPEN {url}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            httpd.shutdown()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
