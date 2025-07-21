
import cv2
import dlib
import numpy as np
from scipy.spatial import distance
from collections import deque
import os


class FaceAnalyzer:
    def __init__(self):
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.face_cascade = self._load_face_cascade()  # 初始化人脸检测器
        self.model = None
        self.emotion_history = deque(maxlen=3)  # 进一步缩短历史队列
        self.confidence_threshold = 0.5  # 添加可调参数
        self.predictor = self._load_shape_predictor()  # 新增面部特征点检测器
        self.eye_close_start_time = None  # 新增闭眼计时器
        self.EMERGENCY_CONFIDENCE = 0.95  # 新增高置信度阈值
        self.EYE_AR_THRESH = 0.25  # 闭眼阈值（可调整）
        self.EYE_CLOSE_DURATION = 2.0  # 必须2秒

    def _load_face_cascade(self):
        """安全加载人脸检测分类器"""
        try:
            # 检查OpenCV内置路径
            cascade_path = os.path.join(
                cv2.data.haarcascades,
                'haarcascade_frontalface_default.xml'
            )

            if not os.path.exists(cascade_path):
                # 尝试备用路径
                cascade_path = 'haarcascade_frontalface_default.xml'
                if not os.path.exists(cascade_path):
                    raise FileNotFoundError("未找到人脸检测模型文件")

            face_cascade = cv2.CascadeClassifier(cascade_path)

            # 验证是否加载成功
            if face_cascade.empty():
                raise ValueError("分类器加载失败")

            print(f"成功加载人脸检测器: {cascade_path}")
            return face_cascade

        except Exception as e:
            print(f"严重错误: {str(e)}")
            raise RuntimeError("无法初始化人脸检测器") from e

    def load_model(self, model_path):
        from tensorflow.keras.models import load_model
        self.model = load_model(model_path)
        # 添加编译参数
        self.model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

    def detect_faces(self, gray):
        return self.face_cascade.detectMultiScale(gray,
                                                  scaleFactor=1.1,
                                                  minNeighbors=5,
                                                  minSize=(48, 48))

    def eye_aspect_ratio(self, eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def _load_shape_predictor(self):
        """加载dlib面部特征点检测器"""
        try:
            predictor_path = "shape_predictor_68_face_landmarks.dat"
            if not os.path.exists(predictor_path):
                raise FileNotFoundError(f"面部特征点模型文件不存在: {predictor_path}")

            predictor = dlib.shape_predictor(predictor_path)
            print("成功加载面部特征点检测器")
            return predictor
        except Exception as e:
            print(f"无法加载面部特征点检测器: {str(e)}")
            return None

    def detect_eyes(self, gray, face_rect):
        try:
            """使用特征点检测眼睛"""
            if self.predictor is None:
                raise RuntimeError("面部特征点检测器未初始化")

            # 将OpenCV矩形转换为dlib矩形
            x, y, w, h = face_rect
            dlib_rect = dlib.rectangle(left=x, top=y, right=x + w, bottom=y + h)

            # 检测特征点
            landmarks = self.predictor(gray, dlib_rect)
            return self._parse_eye_points(landmarks)
            print(f"左眼EAR: {left_ear:.2f}, 右眼EAR: {right_ear:.2f}")
            return (left_ear + right_ear) / 2

        except Exception as e:
            print(f"眼睛检测失败: {str(e)}")
            return 1.0  # 返回默认睁眼值

    def _parse_eye_points(self, landmarks):
        """解析眼睛特征点"""
        # 左眼特征点（索引36-41）
        left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        # 右眼特征点（索引42-47）
        right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
        return left_eye, right_eye

    def check_eyes_closed(self, ear, current_time):
        print(f"[输入] EAR={ear:.2f}, 当前时间={current_time:.2f}, 上次时间={self.eye_close_start_time or 0:.2f}")
        """统一基于时间的检测"""
        if ear < self.EYE_AR_THRESH:
            if self.eye_close_start_time is None:
                self.eye_close_start_time = current_time
                return False
            else:
                duration = current_time - self.eye_close_start_time
                # 添加调试日志
                print(f"[严格检测] 当前持续时间: {duration:.2f}s (阈值: {self.EYE_CLOSE_DURATION}s)")
                return duration >= self.EYE_CLOSE_DURATION
        else:
            self.eye_close_start_time = None
            return False


    def predict_emotion(self, face_img):
        """
        执行表情预测的完整流程，包含：
        1. 输入验证
        2. 图像预处理
        3. 模型预测
        4. 后处理
        """
        # 阶段1：输入验证
        if not self._validate_input(face_img):
            return "Neutral"

        try:
            # 阶段2：图像预处理
            processed_img = self._preprocess_image(face_img)

            # 阶段3：模型预测
            pred = self.model.predict(processed_img, verbose=0)

            # 阶段4：后处理
            return self._postprocess_prediction(pred[0])

        except Exception as e:
            print(f"预测失败: {str(e)}")
            return "Neutral"

    def _validate_input(self, img):
        """验证输入图像的有效性"""
        if img is None:
            print("错误：输入图像为空")
            return False
        if img.size == 0:
            print("错误：输入图像尺寸为0")
            return False
        if len(img.shape) != 2:
            print(f"错误：期望灰度图，实际通道数{len(img.shape)}")
            return False
        return True

    def _preprocess_image(self, img):
        """标准化图像预处理流程"""
        try:
            # 调整尺寸并添加通道维度
            resized = cv2.resize(img, (48, 48))
            expanded = np.expand_dims(resized, axis=(0, -1))
            return expanded.astype('float32') / 255.0
        except cv2.error as e:
            print(f"图像预处理失败: {str(e)}")
            raise

    def _postprocess_prediction(self, probabilities):
        """预测结果后处理（修复显示不一致问题）"""
        # 阶段1：获取原始预测结果
        emotion_idx = np.argmax(probabilities)
        current_emotion = self.emotion_labels[emotion_idx]
        max_prob = probabilities[emotion_idx]

        # 高置信度直接返回（新增）
        if max_prob >= 0.95:
            print("[紧急通道] 高置信度直接返回")
            return current_emotion

        # 调试输出原始预测结果
        print(f"[Debug] 原始预测: {current_emotion} ({max_prob:.2f})")

        # 阶段2：置信度检查（调整阈值）
        confidence_threshold = 0.45  # 降低置信度阈值
        if max_prob < confidence_threshold:
            print(f"低置信度过滤: {current_emotion} ({max_prob:.2f} < {confidence_threshold})")
            current_emotion = "Neutral"

        # 阶段3：更新历史记录
        self.emotion_history.append(current_emotion)

        # 清空历史当检测到突变（新增）
        if len(self.emotion_history) >= 2:
            if self.emotion_history[-1] != self.emotion_history[-2]:
                self.emotion_history.clear()
                self.emotion_history.append(current_emotion)

        # 重构权重计算逻辑
        emotion_weights = {}
        for i, emotion in enumerate(reversed(self.emotion_history)):  # 关键修改：反向遍历
            weight = 4 ** i  # 指数权重 (1, 4, 16...)
            emotion_weights[emotion] = emotion_weights.get(emotion, 0) + weight

        print(f"[修正] 权重分布: {emotion_weights}")

        # 强制最新结果优先（当权重差小于总权重的30%时）
        total_weight = sum(emotion_weights.values())
        final_emotion = max(emotion_weights, key=emotion_weights.get)

        if emotion_weights.get(current_emotion, 0) / total_weight > 0.3:
            return current_emotion

        return final_emotion