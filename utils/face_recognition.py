"""
Module Face Recognition - Trích xuất embedding và so sánh khuôn mặt.
Sử dụng DeepFace + FaceNet512 để trích embedding 512-chiều từ ảnh chân dung.
Tối ưu cho CPU inference.
"""

import numpy as np
import cv2
import os
from scipy.spatial.distance import cosine
import json
import logging
from utils.detector import FaceDetector

logger = logging.getLogger(__name__)

class FaceRecognizer:
    """
    Lớp FaceRecognizer dùng DeepFace (FaceNet512) để trích embedding và so sánh.
    
    Thuộc tính:
        model_name (str): Tên model - sử dụng "FaceNet512" cho accuracy cao
        metric (str): Metric so sánh - "cosine" cho cosine similarity
        detector_backend (str): Backend detection - "opencv" cho CPU tối ưu
    """
    
    def __init__(self, model_name="Facenet512", metric="cosine", detector_backend="opencv"):
        """
        Khởi tạo FaceRecognizer.
        
        Args:
            model_name (str): Model trích embedding. Default: "FaceNet512" (512-dim)
            metric (str): Metric so sánh embedding. Default: "cosine"
            detector_backend (str): Backend detection faces. Default: "opencv" (CPU-friendly)
        """
        self.model_name = model_name
        self.metric = metric
        self.detector_backend = detector_backend
        self.face_detector = FaceDetector() # Tái sử dụng RetinaFace detector
        
        # Lưu cấu hình nhưng không import DeepFace ngay để tránh load TensorFlow khi
        # chỉ import module (lazy load). Việc load model sẽ xảy ra khi thực sự cần.
        self._model_loaded = False
        self._deepface = None

    def extract_embedding(self, img_rgb, enforce_detection=True, detector_backend=None):
        """
        Trích xuất embedding 512-chiều từ 1 ảnh RGB chứa 1 khuôn mặt.
        
        Args:
            img_rgb (np.ndarray): Ảnh RGB shape (H, W, 3)
            enforce_detection (bool): Nếu True, lỗi nếu không nhận ra mặt.
                                     Nếu False, trả về zero vector.
            detector_backend (str): Ghi đè backend detection. Nếu là 'skip' sẽ không detect.
        
        Returns:
            np.ndarray: Embedding vector shape (512,) hoặc None nếu lỗi
            
        Raises:
            ValueError: Nếu ảnh quá mờ/mụa hoặc không nhận ra mặt (khi enforce_detection=True)
        """
        try:
            if img_rgb is None or img_rgb.size == 0:
                logger.warning("Ảnh rỗng")
                return None
                
            # Kiểm tra ảnh quá nhỏ
            h, w = img_rgb.shape[:2]
            if h < 20 or w < 20:
                logger.warning(f"Ảnh quá nhỏ: {w}x{h}")
                return None
            
            # Lazy import DeepFace khi cần (giảm thời gian import module)
            try:
                if self._deepface is None:
                    from deepface import DeepFace
                    self._deepface = DeepFace
            except Exception as e:
                logger.warning(f"Không thể import DeepFace: {e}")
                return None

            # Trích embedding bằng DeepFace
            backend = detector_backend if detector_backend else self.detector_backend
            
            embedding_objs = self._deepface.represent(
                img_rgb,
                model_name=self.model_name,
                detector_backend=backend,
                enforce_detection=enforce_detection
            )
            
            if embedding_objs and len(embedding_objs) > 0:
                # Lấy embedding từ face đầu tiên (lớn nhất/rõ nhất)
                embedding = np.array(embedding_objs[0]["embedding"])
                logger.debug(f"✓ Trích embedding thành công: shape {embedding.shape}")
                return embedding
            else:
                logger.warning("Không tìm thấy khuôn mặt trong ảnh")
                return None
                
        except Exception as e:
            logger.warning(f"Lỗi trích embedding: {str(e)}")
            return None

    def compare_embeddings(self, emb1, emb2, threshold=0.4):
        """
        So sánh 2 embedding vectors bằng cosine similarity.
        
        Args:
            emb1 (np.ndarray): Embedding 1, shape (512,)
            emb2 (np.ndarray): Embedding 2, shape (512,)
            threshold (float): Ngưỡng nhận dạng. Default: 0.4 (cosine similarity)
                              Nếu similarity > threshold → Cùng người
        
        Returns:
            tuple: (is_match: bool, similarity: float)
                   is_match=True nếu similarity > threshold
        """
        try:
            if emb1 is None or emb2 is None:
                return False, 0.0
            
            # Cosine similarity = 1 - cosine_distance
            # Cosine distance trong scipy: 0 = giống nhất, 1 = khác nhất
            distance = cosine(emb1, emb2)
            similarity = 1 - distance
            
            is_match = similarity > threshold
            
            logger.debug(f"So sánh: similarity={similarity:.4f}, threshold={threshold}, match={is_match}")
            return is_match, similarity
            
        except Exception as e:
            logger.warning(f"Lỗi so sánh embedding: {e}")
            return False, 0.0

    def extract_embedding_from_file(self, image_path):
        """
        Trích embedding từ file ảnh.
        
        Args:
            image_path (str): Đường dẫn file ảnh (jpg/png)
        
        Returns:
            np.ndarray: Embedding (512,) hoặc None nếu lỗi
        """
        try:
            if not os.path.exists(image_path):
                logger.warning(f"File ảnh không tồn tại: {image_path}")
                return None
            
            # Đọc ảnh bằng cv2 (BGR)
            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                logger.warning(f"Không thể đọc ảnh: {image_path}")
                return None
            
            # Convert BGR → RGB
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # Trích embedding
            embedding = self.extract_embedding(img_rgb, enforce_detection=True)
            
            return embedding
            
        except Exception as e:
            logger.warning(f"Lỗi đọc + trích embedding từ file: {e}")
            return None

    def get_faces_in_image(self, img_rgb, threshold=0.6):
        """
        Detect tất cả khuôn mặt trong 1 ảnh và trích embedding từng mặt.
        
        Args:
            img_rgb (np.ndarray): Ảnh RGB shape (H, W, 3)
            threshold (float): Ngưỡng confidence để keep detection. Default: 0.6
        
        Returns:
            list: Danh sách dict, mỗi dict chứa:
                  {
                      'bbox': (x1, y1, x2, y2),
                      'embedding': np.ndarray (512,),
                      'confidence': float
                  }
            
        Ghi chú:
            - Nếu lỗi detect hoặc không tìm mặt, trả về []
            - Bỏ qua faces nhỏ hơn 20x20 pixels
        """
        try:
            faces = []
            
            if img_rgb is None or img_rgb.size == 0:
                return faces
            
            h, w = img_rgb.shape[:2]
            
            # Sử dụng FaceDetector (RetinaFace) từ utils/detector.py để đồng bộ
            detections = self.face_detector.detect(img_rgb, conf_thresh=threshold)
            
            for face_obj in detections:
                # face_obj structure từ FaceDetector:
                # {'score': float, 'facial_area': [x1, y1, x2, y2]}
                
                confidence = face_obj.get('score', 0.0)
                facial_area = face_obj.get('facial_area', [0, 0, 0, 0])
                
                x1, y1, x2, y2 = facial_area
                
                # Bỏ qua faces quá nhỏ
                if (x2 - x1) < 20 or (y2 - y1) < 20:
                    continue
                
                # Crop face từ ảnh gốc
                face_crop = img_rgb[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                
                if face_crop.size == 0:
                    continue
                
                # Trích embedding - dùng detector_backend='skip' vì đã crop đúng mặt
                embedding = self.extract_embedding(face_crop, enforce_detection=False, detector_backend='skip')
                
                if embedding is not None:
                    faces.append({
                        'bbox': (x1, y1, x2, y2),
                        'embedding': embedding,
                        'confidence': confidence
                    })
            
            logger.info(f"RetinaFace detect {len(faces)} faces trong ảnh")
            return faces
                
        except Exception as e:
            logger.warning(f"Lỗi trong get_faces_in_image: {e}")
            return []

    @staticmethod
    def embedding_to_json(embedding):
        """
        Convert embedding np.ndarray → JSON string để lưu vào DB.
        
        Args:
            embedding (np.ndarray): Embedding vector (512,)
        
        Returns:
            str: JSON string
        """
        if embedding is None:
            return None
        return json.dumps(embedding.tolist())

    @staticmethod
    def json_to_embedding(json_str):
        """
        Convert JSON string → np.ndarray để so sánh.
        
        Args:
            json_str (str): JSON string
        
        Returns:
            np.ndarray: Embedding vector (512,)
        """
        if json_str is None:
            return None
        try:
            return np.array(json.loads(json_str), dtype=np.float32)
        except Exception as e:
            logger.warning(f"Lỗi parse JSON embedding: {e}")
            return None
