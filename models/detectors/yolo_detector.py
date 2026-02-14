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

    def align_face(self, image: np.ndarray, keypoints: np.ndarray, output_size: int = 112):
        """
        根据5个关键点进行人脸对齐
        """
        desired_pts = np.array(
            [
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041],
            ],
            dtype=np.float32,
        )
        
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
        if self.model is None:
            return []

        if isinstance(image_path_or_array, str):
            image = cv2.imread(image_path_or_array)
            if image is None:
                print(f"Error: Cannot read image {image_path_or_array}")
                return []
        else:
            image = image_path_or_array

        results = self.model.predict(
            image,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )

        aligned_faces = []
        result = results[0]

        if result.boxes is not None and len(result.boxes) > 0:
            for i, box in enumerate(result.boxes.xyxy):
                # 如果有关键点且需要对齐
                if align and result.keypoints is not None and i < len(result.keypoints.xy):
                    try:
                        # YOLO keypoints: (5, 2)
                        keypoints = result.keypoints.xy[i].cpu().numpy()
                        face = self.align_face(image, keypoints, output_size)
                        aligned_faces.append(face)
                    except Exception as e:
                        print(f"Alignment failed: {e}")
                        continue
                else:
                    # 如果不需要对齐或没有关键点，直接裁剪
                    x1, y1, x2, y2 = map(int, box.cpu().numpy())
                    # 边界检查
                    h, w = image.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    face = image[y1:y2, x1:x2]
                    if face.size > 0:
                        face = cv2.resize(face, (output_size, output_size))
                        aligned_faces.append(face)
        
        return aligned_faces

    def detect_faces(self, image_path_or_array, align=True, output_size=112):
        """
        返回详细检测结果: [{box, keypoints, aligned_face, score}, ...]
        """
        if self.model is None:
            return []

        if isinstance(image_path_or_array, str):
            image = cv2.imread(image_path_or_array)
            if image is None: return []
        else:
            image = image_path_or_array

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
        
        return detections
