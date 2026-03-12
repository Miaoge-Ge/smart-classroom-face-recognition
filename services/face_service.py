import torch
import os
import numpy as np
import hashlib
import json
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from core.model_factory import ModelFactory
from core.config_manager import global_config
from models.detectors.yolo_detector import YOLOFaceDetector
import cv2

from core.models import Student, Attendance
from core.database import SessionLocal
from datetime import datetime

try:
    from core.crypto_manager import decrypt_bytes, decrypt_from_b64, encrypt_to_b64
    _crypto_ok = True
except Exception:
    _crypto_ok = False


def _infer_backbone_type_from_weights_path(weights_path: str | None) -> str | None:
    if not weights_path:
        return None
    p = os.path.normpath(str(weights_path)).replace("\\", "/").lower()
    if "/recognition/resnet50/" in p:
        return "resnet50"
    if "/recognition/resnet10/" in p:
        return "resnet10"
    if "/recognition/fastcontextface/" in p:
        return "fastcontextface"
    return None


def _compute_model_sig(
    backbone_type: str,
    embedding_size: int,
    weights_path: str,
    preprocess_cfg,
) -> str:
    p = os.path.abspath(str(weights_path))
    try:
        st = os.stat(p)
        weights_meta = {"path": p, "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}
    except Exception:
        weights_meta = {"path": p}
    prep = preprocess_cfg or {}
    payload = {
        "backbone_type": str(backbone_type),
        "embedding_size": int(embedding_size),
        "weights": weights_meta,
        "preprocess": {
            "input_size": prep.get("input_size"),
            "mean": prep.get("mean"),
            "std": prep.get("std"),
        },
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

class FaceRecognitionService:
    def __init__(self, config=None):
        """
        人脸识别服务
        :param config: Config 对象，默认使用全局配置
        """
        self.cfg = config if config else global_config
        if not self.cfg:
            raise RuntimeError("Configuration not initialized")
            
        self.device = self._get_device()
        print(f"Initializing Face Service on {self.device}...")
        
        # 1. 初始化识别模型
        rec_cfg = dict(self.cfg.recognition or {})
        weights_path = rec_cfg.get("weights_path")
        backbone_type = _infer_backbone_type_from_weights_path(weights_path) or rec_cfg.get("backbone_type")
        embedding_size = rec_cfg.get("embedding_size", 512)
        if not backbone_type:
            raise RuntimeError("recognition.backbone_type 未配置，且无法从 weights_path 推断")
        if not weights_path:
            raise RuntimeError("recognition.weights_path 未配置")
        self.model_sig = _compute_model_sig(
            backbone_type=backbone_type,
            embedding_size=embedding_size,
            weights_path=weights_path,
            preprocess_cfg=self.cfg.preprocess,
        )
        self.model = ModelFactory.create_backbone(
            backbone_type,
            embedding_size,
        )
        self._load_weights(weights_path)
        self.model.to(self.device)
        self.model.eval()
        
        # 相似度阈值
        self.similarity_threshold = rec_cfg.get('similarity_threshold', 0.25)
        
        # 2. 初始化检测模型
        det_cfg = self.cfg.detector
        if det_cfg and det_cfg.get('model_path'):
            det_device = det_cfg.get('device', 'auto')
            if det_device == 'auto':
                det_device = self.device
                
            self.detector = YOLOFaceDetector(
                model_path=det_cfg['model_path'],
                conf_threshold=det_cfg.get('conf_threshold', 0.5),
                device=det_device
            )
        else:
            self.detector = None
            print("Warning: Detector not configured.")
        
        # 3. 初始化预处理
        self.transform = self._build_transforms()
        
        # 4. 已知人脸库 (InMemory Cache for fast retrieval)
        self.known_faces = {}
        self.student_labels = {}
        self._load_faces_from_db()
        
        print("Face Service Initialized Successfully.")

    def _make_student_label(self, name: str | None, student_no: str | None) -> str:
        display_name = (name or "").strip() or "Unknown"
        display_no = (student_no or "").strip()
        return f"{display_name} [{display_no}]" if display_no else display_name

    def upsert_known_face(self, student_id: int | None, name: str | None, feature, student_no: str | None = None):
        if student_id is None or feature is None:
            return
        sid = int(student_id)
        self.known_faces[sid] = feature
        self.student_labels[sid] = self._make_student_label(name, student_no)

    def remove_known_face(self, student_id: int | None):
        if student_id is None:
            return
        sid = int(student_id)
        self.known_faces.pop(sid, None)
        self.student_labels.pop(sid, None)

    def _load_faces_from_db(self):
        """Load registered faces from SQLite to memory"""
        db = SessionLocal()
        try:
            students = db.query(Student).all()
            pending_updates = 0
            for s in students:
                if not s.name:
                    continue
                sig_ok = getattr(s, "face_embedding_model_sig", None) == getattr(self, "model_sig", None)
                if _crypto_ok and sig_ok and getattr(s, "face_embedding_enc", None):
                    try:
                        raw = decrypt_from_b64(s.face_embedding_enc)
                        vec = np.frombuffer(raw, dtype=np.float32)
                        if vec.size > 0:
                            t = torch.from_numpy(vec).to(self.device)
                            self.upsert_known_face(s.student_id, s.name, t, s.student_no)
                            continue
                    except Exception:
                        pass

                if not s.face_image_path:
                    continue

                try:
                    if _crypto_ok and str(s.face_image_path).endswith(".enc") and os.path.exists(s.face_image_path):
                        with open(s.face_image_path, "rb") as f:
                            enc = f.read()
                        plain = decrypt_bytes(enc)
                        nparr = np.frombuffer(plain, np.uint8)
                        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if image is None:
                            continue
                        feats = self.process_image(image)
                    else:
                        if not os.path.exists(s.face_image_path):
                            continue
                        feats = self.process_image(s.face_image_path)

                    if feats:
                        self.upsert_known_face(s.student_id, s.name, feats[0], s.student_no)
                        if _crypto_ok and getattr(self, "model_sig", None):
                            try:
                                raw = feats[0].detach().cpu().numpy().astype(np.float32).tobytes()
                                s.face_embedding_enc = encrypt_to_b64(raw)
                                s.face_embedding_model_sig = self.model_sig
                                db.add(s)
                                pending_updates += 1
                                if pending_updates >= 50:
                                    db.commit()
                                    pending_updates = 0
                            except Exception:
                                db.rollback()
                except Exception as e:
                    print(f"Error loading face for {s.name}: {e}")
            if pending_updates:
                db.commit()
            print(f"Loaded {len(self.known_faces)} faces from database.")
        finally:
            db.close()

    def register_person(self, name, image_input, student_no=None):
        """
        注册已知人脸到数据库
        :param name: 姓名
        :param image_input: 图片路径
        """
        feats = self.process_image(image_input)
        if feats:
            db = SessionLocal()
            try:
                # Check if exists
                query = db.query(Student)
                if student_no:
                    student = query.filter(Student.student_no == student_no).first()
                else:
                    student = query.filter(Student.name == name).first()
                if not student:
                    student = Student(name=name, student_no=student_no or name, face_image_path=image_input)
                    db.add(student)
                    db.flush()
                else:
                    student.face_image_path = image_input # Update photo
                    if student_no:
                        student.student_no = student_no

                self.upsert_known_face(student.student_id, student.name, feats[0], student.student_no)
                db.commit()
                print(f"Registered user: {name} in DB")
                return True
            except Exception as e:
                print(f"DB Error: {e}")
                db.rollback()
                return False
            finally:
                db.close()
                
        print(f"Failed to register user: {name} (No face detected)")
        return False

    def recognize_frame(self, frame):
        """
        处理视频帧: 检测 -> 识别 -> 匹配
        :param frame: numpy array (BGR), e.g. from cv2.imread or cap.read()
        :return: List of dict {'box': [x1,y1,x2,y2], 'name': str, 'score': float}
        """
        if self.detector is None:
            return []
            
        # 1. Detect (使用新添加的 detect_faces 获取 box 和 aligned_face)
        # 注意: 需要确保 YOLOFaceDetector 已经添加了 detect_faces 方法
        if not hasattr(self.detector, 'detect_faces'):
             print("Error: Detector does not support detect_faces")
             return []
             
        detections = self.detector.detect_faces(frame)
        results = []
        
        for det in detections:
            box = det['box']
            aligned_face = det['aligned_face']
            
            if aligned_face is None:
                continue

            # 2. Extract Feature
            # aligned_face is BGR numpy array
            face_img_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(face_img_rgb)
            feat = self.extract_feature(pil_img)
            
            # 3. Match
            best_name = "Unknown"
            best_student_id = None
            best_score = 0.0
            
            if feat is not None and self.known_faces:
                for student_id, known_feat in self.known_faces.items():
                    # is_same_person 返回 (bool, score)
                    is_same, score = self.is_same_person(feat, known_feat)
                    if score > best_score:
                        best_score = score
                        if is_same:
                            best_student_id = int(student_id)
                            best_name = self.student_labels.get(best_student_id, "Unknown")
            
            # 如果最高分没过阈值，依然是 Unknown
            if best_score < self.similarity_threshold:
                best_name = "Unknown"
                best_student_id = None

            results.append({
                "box": box,
                "name": best_name,
                "student_id": best_student_id,
                "score": float(best_score)
            })
            
        return results

    def _get_device(self):
        # 优先从 recognition 配置中读取 device
        device_str = self.cfg.recognition.get('device', 'auto')
        if device_str == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device_str

    def _load_weights(self, weight_path):
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Weight file not found: {weight_path}")
            
        try:
            checkpoint = torch.load(weight_path, map_location=self.device)
            
            # 尝试不同的 key 获取 state_dict
            if 'model' in checkpoint:
                # 用户确认保存时使用的 key 是 'model'
                state_dict = checkpoint['model']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # 处理 'module.' 和 'backbone.' 前缀
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace('module.', '')
                if name.startswith('backbone.'):
                    name = name.replace('backbone.', '')
                new_state_dict[name] = v
                
            missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
            
            if missing:
                print(f"\n[Weight Loading Debug] Missing keys ({len(missing)}): {missing[:5]} ...")
                # 过滤掉 fc 和 aux_head 等不重要的层，如果 conv 层缺失则是大问题
                critical_missing = [k for k in missing if 'conv' in k or 'bn' in k]
                if critical_missing:
                    print(f"CRITICAL WARNING: 核心权重缺失! {critical_missing[:5]} ...")
            
            if unexpected:
                # 只要不是 backbone 里的层多出来了，一般没事 (比如 head.weight)
                pass
                # print(f"[Weight Loading Debug] Unexpected keys: {unexpected[:5]}")
        except Exception as e:
            raise RuntimeError(f"Failed to load weights: {e}")

    def _build_transforms(self):
        prep_cfg = self.cfg.preprocess
        h, w = prep_cfg['input_size']
        
        return transforms.Compose([
            transforms.Resize((h, w), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=prep_cfg['mean'], std=prep_cfg['std'])
        ])

    def extract_feature(self, img_input):
        """
        提取特征 (底层接口): 假设输入已经是 112x112 的对齐人脸
        Returns:
            embedding: tensor, shape [512] (on device)
        """
        # 1. 预处理
        tensor = self._preprocess(img_input)
        if tensor is None:
            return None
            
        # 2. 推理
        with torch.no_grad():
            embedding = self.model(tensor)
            # 3. 归一化 (L2 Normalize)
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            
        # 注意：这里不再转 numpy，直接返回 tensor 以便进行高效的矩阵运算
        return embedding.squeeze(0) # [1, 512] -> [512]

    def process_image(self, image_path):
        """
        高层接口: 输入原始大图 -> 检测 -> 对齐 -> 识别
        返回: List of feature tensors
        """
        if self.detector is None:
            print("Error: Detector not initialized. Please configure 'detector' in config.yaml")
            return []
            
        # 1. 检测并对齐
        # 返回的是 numpy array (BGR) 列表，已经是 112x112
        aligned_faces = self.detector.detect_and_align(image_path)
        
        features = []
        for face_img in aligned_faces:
            # 2. 转换颜色空间 BGR -> RGB
            face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(face_img_rgb)
            
            # 3. 提取特征
            feat = self.extract_feature(pil_img)
            if feat is not None:
                features.append(feat)
                
        return features

    def _preprocess(self, img_input):
        if isinstance(img_input, str):
            if not os.path.exists(img_input):
                print(f"Error: Image not found {img_input}")
                return None
            img = Image.open(img_input).convert('RGB')
        elif isinstance(img_input, Image.Image):
            img = img_input.convert('RGB')
        else:
            print("Error: Invalid input type")
            return None
            
        return self.transform(img).unsqueeze(0).to(self.device)

    def compute_similarity(self, feat1, feat2):
        """
        计算余弦相似度 (Torch 版本)
        Args:
            feat1: tensor [512]
            feat2: tensor [512]
        """
        if feat1 is None or feat2 is None:
            return 0.0
            
        # 确保输入是 tensor 并且在同一个 device 上
        if not isinstance(feat1, torch.Tensor):
            feat1 = torch.tensor(feat1, device=self.device)
        if not isinstance(feat2, torch.Tensor):
            feat2 = torch.tensor(feat2, device=self.device)
            
        # 已经在 extract_feature 做过 F.normalize 了，这里直接点积即可
        # 如果不放心，可以再做一次，但会浪费一点计算量
        # 为了严格复刻你的逻辑，我们这里假设输入已经是 normalized 的
        
        return torch.dot(feat1, feat2).item()

    def is_same_person(self, feat1, feat2, threshold=None):
        """
        判断是否为同一个人
        :param threshold: 可选，覆盖默认阈值
        :return: (bool, score)
        """
        score = self.compute_similarity(feat1, feat2)
        thresh = threshold if threshold is not None else self.similarity_threshold
        return score > thresh, score
