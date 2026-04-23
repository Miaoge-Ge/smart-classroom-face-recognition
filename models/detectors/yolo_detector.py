import cv2
import torch
import numpy as np
import os
from ultralytics import YOLO

class YOLOFaceDetector:
    def __init__(self, model_path, conf_threshold=0.1, device=None):
        self.conf_threshold = conf_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        if not os.path.exists(model_path):
            print(f"Warning: YOLO model not found at {model_path}")
        
        print(f"Loading YOLO Face Detector: {model_path} on {self.device}")
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.model = None

    @staticmethod
    def _load_image(image_path_or_array):
        if isinstance(image_path_or_array, str):
            image = cv2.imread(image_path_or_array)
            if image is None:
                print(f"Error: Cannot read image {image_path_or_array}")
            return image
        return image_path_or_array

    @staticmethod
    def _priority_score(det_info, image_shape):
        if image_shape is None:
            return float(det_info.get("score", 0.0))

        h, w = image_shape[:2]
        x1, y1, x2, y2 = det_info["box"]
        box_w = max(1.0, float(x2 - x1))
        box_h = max(1.0, float(y2 - y1))
        area = box_w * box_h

        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        frame_center_x = w / 2.0
        frame_center_y = h / 2.0
        center_distance = np.hypot(center_x - frame_center_x, center_y - frame_center_y)
        frame_radius = max(np.hypot(frame_center_x, frame_center_y), 1.0)
        center_weight = max(0.35, 1.0 - (center_distance / frame_radius))

        return area * center_weight * max(0.1, float(det_info.get("score", 0.0)))

    def align_face(self, image: np.ndarray, keypoints: np.ndarray, output_size: int = 112):
        """
        根据5个关键点进行人脸对齐
        """
        # 标准 InsightFace 顺序 (112x112):
        standard_pts = np.array([
            [38.2946, 51.6963], # 左眼
            [73.5318, 51.5014], # 右眼
            [56.0252, 71.7366], # 鼻子
            [41.5493, 92.3655], # 左嘴角
            [70.7299, 92.2041]  # 右嘴角
        ], dtype=np.float32)

        detected_pts = keypoints.astype(np.float32)
        
        # 兼容 YOLO Pose (17点) 和 YOLO Face (5点)
        if detected_pts.shape[0] == 17:
            # COCO Keypoints: 0:Nose, 1:L-Eye, 2:R-Eye
            src_pts = detected_pts[[1, 2, 0]]
            dst_pts = standard_pts[[0, 1, 2]]
            transform_matrix, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        elif detected_pts.shape[0] >= 5:
            reordered_pts = detected_pts[[3, 4, 2, 0, 1]] if detected_pts.shape[0] == 5 else detected_pts[:5]
            transform_matrix, _ = cv2.estimateAffinePartial2D(reordered_pts, standard_pts)
        else:
            # 如果关键点不足，直接返回原图像（不做对齐）
            h, w = image.shape[:2]
            return cv2.resize(image, (output_size, output_size))

        aligned_face = cv2.warpAffine(
            image,
            transform_matrix,
            (output_size, output_size),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=[0, 0, 0],
        )
        return aligned_face

    def detect_and_align(self, image_path_or_array, align=True, output_size=112):
        """
        核心接口：输入图片，输出检测并对齐后的人脸列表
        :return: list of aligned_face (numpy array BGR)
        """
        detections = self.detect_faces(image_path_or_array, align=align, output_size=output_size)
        return [det["aligned_face"] for det in detections if det.get("aligned_face") is not None]

    def detect_faces(self, image_path_or_array, align=True, output_size=112):
        """
        返回详细检测结果: [{box, keypoints, aligned_face, score}, ...]
        """
        if self.model is None:
            return []

        image = self._load_image(image_path_or_array)
        if image is None:
            return []

        results = self.model.predict(
            image,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )

        detections = []
        result = results[0]

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            
            for i, box in enumerate(boxes):
                det_info = {
                    "box": box.astype(int).tolist(),
                    "score": float(confs[i]),
                    "keypoints": None,
                    "aligned_face": None,
                    "yaw": 0.0,
                    "pitch": 0.0,
                    "roll": 0.0
                }

                # Aligned Face
                if align and result.keypoints is not None and i < len(result.keypoints.xy):
                    try:
                        kpts = result.keypoints.xy[i].cpu().numpy()
                        det_info["keypoints"] = kpts.tolist()
                        det_info["aligned_face"] = self.align_face(image, kpts, output_size)
                    except Exception:
                        pass
                
                # Fallback crop if alignment failed or not requested
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

        detections.sort(
            key=lambda det: self._priority_score(det, image.shape if image is not None else None),
            reverse=True,
        )
        
        return detections
