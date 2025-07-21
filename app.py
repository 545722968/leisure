import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from detection_utils import FaceAnalyzer
from audio_utils import AudioPlayer, VoiceAlert
import time
from collections import deque


class DriverMonitorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.recent_eye_states = None
        self.title("驾驶员状态监测系统")
        self.geometry("1200x800")
        self._setup_ui()
        self._init_system()
        self.protocol("WM_DELETE_WINDOW", self._safe_exit)
        self.eye_lock = threading.Lock()  # 新增线程锁
        self.eye_close_frames = 0
        self.EYE_CLOSE_CONSEC_FRAMES = 10  # 需连续10帧中8帧闭眼

    def _setup_ui(self):
        """初始化用户界面"""
        # 主容器
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # 视频显示区域
        self.video_frame = ttk.LabelFrame(self.main_frame, text="实时画面")
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.canvas = tk.Canvas(self.video_frame, bg='#2c3e50', width=800, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 控制面板
        self.control_frame = ttk.LabelFrame(self.main_frame, text="控制面板")
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)

        """创建状态面板组件"""
        self.status_frame = ttk.LabelFrame(self.control_frame, text="系统状态")
        self.status_frame.pack(fill=tk.X, pady=10)

        # 功能按钮
        btn_config = [
            ("打开摄像头", self.open_camera),
            ("打开图片", self.open_image),
            ("打开视频", self.open_video)
        ]
        for text, cmd in btn_config:
            btn = ttk.Button(self.control_frame, text=text, command=cmd)
            btn.pack(fill=tk.X, pady=5)

        # 状态显示
        self.status_frame = ttk.LabelFrame(self.control_frame, text="系统状态")
        self.status_frame.pack(fill=tk.X, pady=10)

        self.lbl_emotion = ttk.Label(self.status_frame, text="表情: 等待检测...")
        self.lbl_emotion.pack(anchor=tk.W)

        self.lbl_eyes = ttk.Label(self.status_frame, text="眼睛: 正常")
        self.lbl_eyes.pack(anchor=tk.W)

        self.lbl_fps = ttk.Label(self.status_frame, text="帧率: 0 FPS")
        self.lbl_fps.pack(anchor=tk.W)

        # 添加音乐冷却标签
        self.lbl_music_cd = ttk.Label(self.status_frame, text="音乐冷却: 就绪")
        self.lbl_music_cd.pack(anchor=tk.W)  # 确保使用self.前缀

    def _init_system(self):
        """初始化系统组件"""
        self.vid = None
        self.photo = None
        self.process_fps = 15  # 处理帧率限制
        self.last_process = 0

        try:
            # 初始化人脸检测器
            self.analyzer = FaceAnalyzer()  # 此处会自动加载分类器

            # 加载表情识别模型
            self.analyzer.load_model('fer2013_plus_CNNmodel.h5')

            # 验证组件状态
            if self.analyzer.face_cascade is None:
                raise RuntimeError("人脸检测器未正确初始化")
            if self.analyzer.model is None:
                raise RuntimeError("表情模型未正确加载")

        except Exception as e:
            error_msg = (
                f"系统初始化失败: {str(e)}\n"
                "可能原因:\n"
                "1. 缺少haarcascade_frontalface_default.xml文件\n"
                "2. OpenCV安装不完整\n"
                "3. 模型文件缺失或损坏"
            )
            messagebox.showerror("致命错误", error_msg)
            self.destroy()

        # 初始化音频
        self.audio_player = AudioPlayer()
        self.voice_alert = VoiceAlert()

        # 历史缓冲区
        self.emotion_history = deque(maxlen=15)
        self.last_alert = {'music': 0, 'voice': 0}

    def open_camera(self):
        """打开摄像头"""
        self._release_resources()
        try:
            self.vid = cv2.VideoCapture(0)
            self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self._update_frame()
        except Exception as e:
            messagebox.showerror("摄像头错误", f"无法打开摄像头:\n{str(e)}")

    def open_image(self):
        """改进的图片打开方法"""
        self._release_resources()  # 调用统一释放
        path = filedialog.askopenfilename(
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp")]
        )
        if path:
            self._release_resources()
            try:
                # 验证文件可读性
                with open(path, 'rb') as f:
                    pass
                # 使用imdecode处理中文路径
                img_data = np.fromfile(path, dtype=np.uint8)
                frame = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError("不支持的图片格式")
                self._process_and_show(frame)
            except Exception as e:
                messagebox.showerror("图片错误", f"无法读取图片:\n{str(e)}")

    def open_video(self):
        """改进的视频打开方法"""
        self._release_resources()  # 调用统一释放
        path = filedialog.askopenfilename(
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv")]
        )
        if path:
            self._release_resources()
            try:
                # 使用CAP_FFMPEG处理更多格式
                self.vid = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
                if not self.vid.isOpened():
                    raise ValueError("不支持的视频格式或编码")
                self._update_frame()
            except Exception as e:
                messagebox.showerror("视频错误", f"无法打开视频:\n{str(e)}")

    def _process_and_show(self, frame):
        start_time = time.time()
        current_emotion = "Neutral" # 初始化默认值
        ear = 1.0   # 默认睁眼状态

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 颜色空间转换
            faces = self.analyzer.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(48, 48)
            )   # 人脸检测

            for (x, y, w, h) in faces:
                # 绘制绿色矩形框（BGR颜色空间）
                cv2.rectangle(
                    img=frame,
                    pt1=(x, y),
                    pt2=(x + w, y + h),
                    color=(0, 255, 0),  # 绿色
                    thickness=2,  # 线宽
                    lineType=cv2.LINE_AA
                )
                # 表情识别
                face_img = gray[y:y + h, x:x + w]
                emotion = self.analyzer.predict_emotion(face_img)
                current_emotion = emotion  # 更新当前表情

                # 在检测框上方添加文字
                cv2.putText(
                    img=frame,
                    text=emotion,
                    org=(x, y - 10),  # 文字位置
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.9,  # 字体大小
                    color=(0, 255, 0),  # 绿色
                    thickness=2,
                    lineType=cv2.LINE_AA
                )

                # 眼睛检测
                # left_eye, right_eye = self.analyzer.detect_eyes(gray, (x, y, w, h))

                # 改进后的眼睛检测调用
                eye_result = self.analyzer.detect_eyes(gray, (x, y, w, h))

                # 安全解包
                if eye_result and all(eye_result):
                    left_eye, right_eye = eye_result
                else:
                    left_eye, right_eye = [], []  # 默认值
                    print("获得无效的眼睛坐标")

                ear = (self.analyzer.eye_aspect_ratio(left_eye) +
                       self.analyzer.eye_aspect_ratio(right_eye)) / 2
                # 传递当前帧处理时间  触发警报
                self._check_alerts(
                    emotion=current_emotion,
                    ear=ear,
                    current_time=time.time()  # 关键修复
                )

            fps = 1 / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
            self._update_status(
                emotion=current_emotion,  # 使用最终处理后的结果
                eye_state=ear < self.analyzer.EYE_AR_THRESH,
                fps=fps
            )   # 更新UI状态

        except Exception as e:
            print(f"处理错误: {str(e)}")

        self._show_frame(frame)

    def _check_alerts(self, emotion, ear, current_time):
        with self.eye_lock:  # 加锁
            should_alert = self.analyzer.check_eyes_closed(ear, current_time)

        if should_alert:
            print(f"[触发逻辑] 符合条件，当前时间差: {current_time - self.analyzer.eye_close_start_time:.2f}s")
            self.voice_alert.play_alert('alert.wav')
            self.analyzer.eye_close_start_time = None  # 重置状态
        """检查警报条件"""
        current_time = time.time()
        # 闭眼警报
        if self.analyzer.check_eyes_closed(ear, current_time):
            if not self.audio_player.playing:  # 防止重复触发
                self.voice_alert.play_alert('alert.wav')
                print(f"[警报] {time.strftime('%Y-%m-%d %H:%M:%S')} 检测到持续闭眼")

        # 情绪警报（新增冷却时间显示）
        if emotion in ['Angry', 'Sad']:
            time_since_last = current_time - self.last_alert['music']
            if time_since_last > 2:
                self.audio_player.play_music('calm_music.wav')
                self.last_alert['music'] = current_time
                # 更新冷却显示（添加存在性检查）
                if hasattr(self, 'lbl_music_cd'):
                    remaining = 2 - int(time_since_last)
                    self.lbl_music_cd.config(
                        text=f"音乐冷却: {max(remaining, 0)}s",
                        foreground="red" if remaining > 0 else "green"
                    )

    def _check_eyes(self, ear):
        # 使用队列保存最近帧状态
        if ear < self.analyzer.adaptive_ear_threshold:
            self.recent_eye_states.append(1)  # 1表示闭眼
        else:
            self.recent_eye_states.append(0)  # 0表示睁眼

        # 保持队列长度
        if len(self.recent_eye_states) > self.EYE_CLOSE_CONSEC_FRAMES:
            self.recent_eye_states.pop(0)

        # 触发条件：最近10帧中至少有8帧闭眼
        if sum(self.recent_eye_states) >= 8:
            return True
        return False

    def _create_control_panel(self):
        ttk.Button(
            self.control_frame,
            text="眼睛校准",
            command=self.start_ear_calibration
        ).pack(pady=5)

    def start_ear_calibration(self):
        """引导用户完成校准"""
        messagebox.showinfo("校准提示", "请保持正常睁眼状态看向摄像头5秒")
        self.calibrating = True
        self.after(5000, self.end_calibration)  # 5秒后结束校准

    def end_calibration(self):
        self.calibrating = False
        if self.analyzer.calibrate_ear_threshold():
            messagebox.showinfo("校准完成", f"已根据您的眼睛特征调整灵敏度")
        else:
            messagebox.showerror("校准失败", "请保持正对摄像头并睁开双眼")

    def _show_frame(self, frame):
        """显示帧到Canvas"""
        try:
            # 将OpenCV的BGR格式转换为RGB
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = img.resize((800, 600), Image.Resampling.LANCZOS)

            # 更新Canvas显示
            self.photo = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        except Exception as e:
            print(f"显示错误: {str(e)}")

    def _update_frame(self):
        start_time = time.perf_counter()
        """更新视频帧"""
        if self.vid and self.vid.isOpened():
            try:
                ret, frame = self.vid.read()
                if ret:
                    # 控制处理频率
                    if time.time() - self.last_process > 1 / self.process_fps:
                        self._process_and_show(frame)
                        self.last_process = time.time()
                else:
                    self._release_resources()
            except Exception as e:
                print(f"捕获错误: {str(e)}")
                self._release_resources()

        self.after(50, self._update_frame)
        processing_time = time.perf_counter() - start_time
        self.current_fps = 1.0 / processing_time if processing_time > 0 else 0
        self.analyzer.fps = self.current_fps  # 传递帧率给检测器

    def _update_status(self, emotion, eye_state, fps):
        """实时更新状态面板"""
        # 表情状态
        emotion_color = "#2ecc71" if emotion == "Happy" else "#e74c3c"
        self.lbl_emotion.config(
            text=f"表情: {emotion}",
            foreground=emotion_color
        )

        # 眼睛状态
        eye_color = "#e67e22" if eye_state else "#2c3e50"
        self.lbl_eyes.config(
            text=f"眼睛: {'闭合' if eye_state else '正常'}",
            foreground=eye_color
        )

        # 帧率显示
        self.lbl_fps.config(text=f"帧率: {fps:.1f} FPS")

        # 强制刷新UI
        self.status_frame.update_idletasks()

    def _release_resources(self):
        """统一资源释放方法"""
        # 停止所有音频
        self.audio_player.stop_music()

        # 释放视频资源
        if self.vid:
            self.vid.release()
            self.vid = None
        cv2.destroyAllWindows()
        self.canvas.delete("all")

    def _safe_exit(self):
        """安全退出程序"""
        self._release_resources()
        self.destroy()


if __name__ == "__main__":
    app = DriverMonitorApp()
    app.mainloop()