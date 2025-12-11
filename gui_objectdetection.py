import os
import sys
import time
import cv2
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt, QTimer, QCoreApplication
from PyQt5.QtGui import QPixmap, QImage, QIntValidator
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QVBoxLayout,
    QGroupBox,
    QSlider,
    QPushButton,
)

# 从user_code导入实际的AlgorithmArrangement类
from user_code import AlgorithmArrangement

# 整合常量到类属性，消除全局变量冗余
class Ui_Dialog(object):
    MAX_CONTROL_WIDTH = 355
    DEFAULT_CAM_FPS = 25
    DEFAULT_OUTPUT_FPS = 25

    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.setWindowModality(QtCore.Qt.ApplicationModal)
        Dialog.resize(1000, 500)
        self.verticalLayout_6 = QVBoxLayout(Dialog)
        self.verticalLayout_5 = QVBoxLayout()
        self.horizontalLayout_4 = QHBoxLayout()
        self.verticalLayout_2 = QVBoxLayout()

        # --- 1. 模型文件选择 ---
        self.groupBox_model = QGroupBox(Dialog)
        self.groupBox_model.setMaximumSize(QtCore.QSize(self.MAX_CONTROL_WIDTH, 100))
        self.groupBox_model.setObjectName("groupBox_model")
        self.verticalLayout_model = QVBoxLayout(self.groupBox_model)
        self.verticalLayout_model.setSpacing(5)
        self.verticalLayout_model.setContentsMargins(10, 5, 10, 5)

        # 模型文件输入行
        self.horizontalLayout = QHBoxLayout()
        self.label = QLabel(self.groupBox_model)
        self.label.setMinimumWidth(80)
        self.horizontalLayout.addWidget(self.label)
        self.weight_path = QLineEdit(self.groupBox_model)
        self.horizontalLayout.addWidget(self.weight_path)
        self.btn_open_weights = QPushButton(self.groupBox_model)
        self.horizontalLayout.addWidget(self.btn_open_weights)
        self.verticalLayout_model.addLayout(self.horizontalLayout)
        self.verticalLayout_2.addWidget(self.groupBox_model)

        # --- 2. 数据源选择 ---
        self.groupBox_source = QGroupBox(Dialog)
        self.groupBox_source.setMaximumSize(QtCore.QSize(self.MAX_CONTROL_WIDTH, 200))
        self.groupBox_source.setObjectName("groupBox_source")
        self.verticalLayout_source = QVBoxLayout(self.groupBox_source)
        self.verticalLayout_source.setSpacing(5)
        self.verticalLayout_source.setContentsMargins(10, 5, 10, 5)

        # 复用sizePolicy，消除重复创建
        self.label_size_policy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        # 图像文件输入
        self.horizontalLayout_2 = QHBoxLayout()
        self.label_2 = QLabel(self.groupBox_source)
        self.label_2.setMinimumWidth(80)
        self.horizontalLayout_2.addWidget(self.label_2)
        self.image_path = QLineEdit(self.groupBox_source)
        self.horizontalLayout_2.addWidget(self.image_path)
        self.btn_open_image = QPushButton(self.groupBox_source)
        self.horizontalLayout_2.addWidget(self.btn_open_image)
        self.verticalLayout_source.addLayout(self.horizontalLayout_2)

        # 视频文件输入
        self.horizontalLayout_3 = QHBoxLayout()
        self.label_3 = QLabel(self.groupBox_source)
        self.label_3.setMinimumWidth(80)
        self.horizontalLayout_3.addWidget(self.label_3)
        self.video_path = QLineEdit(self.groupBox_source)
        self.horizontalLayout_3.addWidget(self.video_path)
        self.btn_open_video = QPushButton(self.groupBox_source)
        self.horizontalLayout_3.addWidget(self.btn_open_video)
        self.verticalLayout_source.addLayout(self.horizontalLayout_3)

        # RTSP 流输入
        self.horizontalLayout_rtsp = QHBoxLayout()
        self.label_rtsp = QLabel(self.groupBox_source)
        self.label_rtsp.setMinimumWidth(80)
        self.horizontalLayout_rtsp.addWidget(self.label_rtsp)
        self.input_rtsp = QLineEdit(self.groupBox_source)
        self.input_rtsp.setText("rtsp://admin:jndxc301@192.168.2.150")
        self.horizontalLayout_rtsp.addWidget(self.input_rtsp)
        self.btn_open_rtsp = QPushButton(self.groupBox_source)
        self.horizontalLayout_rtsp.addWidget(self.btn_open_rtsp)
        self.verticalLayout_source.addLayout(self.horizontalLayout_rtsp)

        # 摄像头按钮行
        self.horizontalLayout_cam = QHBoxLayout()
        self.label_cam_tip = QLabel(self.groupBox_source)
        self.label_cam_tip.setMinimumWidth(80)
        self.label_cam_tip.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.horizontalLayout_cam.addWidget(self.label_cam_tip)
        self.horizontalLayout_cam.addStretch(1)
        self.btn_toggle_cam = QPushButton(self.groupBox_source)
        self.btn_toggle_cam.setFixedWidth(249)
        self.horizontalLayout_cam.addWidget(self.btn_toggle_cam)
        self.verticalLayout_source.addLayout(self.horizontalLayout_cam)
        self.verticalLayout_2.addWidget(self.groupBox_source)

        # --- 3. 控制参数 GroupBox ---
        self.groupBox_control = QGroupBox(Dialog)
        self.groupBox_control.setMaximumSize(QtCore.QSize(self.MAX_CONTROL_WIDTH, 90))
        self.groupBox_control.setObjectName("groupBox_control")
        self.verticalLayout_control = QVBoxLayout(self.groupBox_control)
        self.verticalLayout_control.setSpacing(5)
        self.verticalLayout_control.setContentsMargins(10, 5, 10, 5)

        # 帧率设置行
        self.horizontalLayout_fps = QHBoxLayout()
        self.horizontalLayout_fps.setSpacing(5)
        self.label_cam_fps = QLabel(self.groupBox_control)
        self.cam_fps_input = QLineEdit(self.groupBox_control)
        self.cam_fps_input.setValidator(QIntValidator(1, 60))
        self.cam_fps_input.setText(str(self.DEFAULT_CAM_FPS))
        self.cam_fps_input.setMaximumWidth(80)

        self.label_output_fps = QLabel(self.groupBox_control)
        self.output_fps_input = QLineEdit(self.groupBox_control)
        self.output_fps_input.setValidator(QIntValidator(1, 60))
        self.output_fps_input.setText(str(self.DEFAULT_OUTPUT_FPS))
        self.output_fps_input.setMaximumWidth(80)

        self.horizontalLayout_fps.addWidget(self.label_cam_fps)
        self.horizontalLayout_fps.addWidget(self.cam_fps_input)
        self.horizontalLayout_fps.addStretch(1)
        self.horizontalLayout_fps.addWidget(self.label_output_fps)
        self.horizontalLayout_fps.addWidget(self.output_fps_input)
        self.verticalLayout_control.addLayout(self.horizontalLayout_fps)

        self.btn_start = QPushButton(self.groupBox_control)
        self.btn_start.setSizePolicy(self.label_size_policy)
        self.btn_start.setMinimumSize(QtCore.QSize(333, 25))
        self.btn_start.setMaximumSize(QtCore.QSize(333, 25))
        self.btn_start.setObjectName("btn_start")
        self.verticalLayout_control.addWidget(self.btn_start)
        self.verticalLayout_2.addWidget(self.groupBox_control)

        # 进度条
        self.video_slider = QSlider(Dialog)
        self.video_slider.setOrientation(QtCore.Qt.Horizontal)
        self.verticalLayout_2.addWidget(self.video_slider)

        # 原始图像 GroupBox
        self.groupBox_2 = QGroupBox(Dialog)
        self.groupBox_2.setMaximumSize(QtCore.QSize(self.MAX_CONTROL_WIDTH, 270))
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox_2)
        self.image_in = QLabel(self.groupBox_2)
        self.image_in.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_in.setMinimumSize(QtCore.QSize(320, 234))
        self.image_in.setMaximumSize(QtCore.QSize(320, 240))
        self.image_in.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.image_in.setAlignment(QtCore.Qt.AlignCenter)
        self.gridLayout.addWidget(self.image_in, 0, 0, 1, 1)
        self.verticalLayout_2.addWidget(self.groupBox_2)

        # 伸缩比例
        self.verticalLayout_2.setStretch(0, 1)
        self.verticalLayout_2.setStretch(1, 2)
        self.verticalLayout_2.setStretch(2, 1)
        self.verticalLayout_2.setStretch(3, 0)
        self.verticalLayout_2.setStretch(4, 3)

        self.horizontalLayout_4.addLayout(self.verticalLayout_2)

        # 右侧输出图像 GroupBox
        self.groupBox_3 = QGroupBox(Dialog)
        self.groupBox_3.setMaximumSize(QtCore.QSize(800, 600))
        self.groupBox_3.setObjectName("groupBox_3")
        self.verticalLayout_3 = QVBoxLayout(self.groupBox_3)
        self.image_out = QLabel(self.groupBox_3)
        self.image_out.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_out.setMinimumSize(QtCore.QSize(691, 0))
        self.image_out.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.image_out.setAlignment(QtCore.Qt.AlignCenter)
        self.verticalLayout_3.addWidget(self.image_out)
        self.horizontalLayout_4.addWidget(self.groupBox_3)

        self.horizontalLayout_4.setStretch(0, 1)
        self.horizontalLayout_4.setStretch(1, 2)
        self.verticalLayout_5.addLayout(self.horizontalLayout_4)
        self.verticalLayout_5.setStretch(0, 1)
        self.verticalLayout_6.addLayout(self.verticalLayout_5)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QCoreApplication.translate
        Dialog.setWindowTitle(
            _translate(
                "Dialog",
                "YOLO检测算法调试助手,若使用网络流,请按这种格式填信息rtsp://用户名:密码@ip",
            )
        )

        self.groupBox_model.setTitle(_translate("Dialog", "模型文件选择"))
        self.groupBox_source.setTitle(_translate("Dialog", "数据源选择(选择一个)"))
        self.groupBox_control.setTitle(_translate("Dialog", "运行控制"))

        self.label.setText(_translate("Dialog", "模型文件"))
        self.btn_open_weights.setText(_translate("Dialog", "选择模型"))
        self.label_2.setText(_translate("Dialog", "本地图像"))
        self.btn_open_image.setText(_translate("Dialog", "选择图片"))
        self.label_3.setText(_translate("Dialog", "本地视频"))
        self.btn_open_video.setText(_translate("Dialog", "选择视频"))
        self.label_rtsp.setText(_translate("Dialog", "网络流(RTSP)"))
        self.btn_open_rtsp.setText(_translate("Dialog", "打开"))
        self.label_cam_tip.setText(_translate("Dialog", f"前置摄像头"))
        self.btn_toggle_cam.setText(_translate("Dialog", "开启摄像头"))
        self.label_cam_fps.setText(_translate("Dialog", "视频源帧率"))
        self.label_output_fps.setText(_translate("Dialog", "输出帧率"))
        self.btn_start.setText(_translate("Dialog", "开始运行"))

        self.groupBox_2.setTitle(_translate("Dialog", "原始图像"))
        self.image_in.setText(_translate("Dialog", "原始图像"))
        self.groupBox_3.setTitle(_translate("Dialog", "输出图像"))
        self.image_out.setText(_translate("Dialog", "识别结果"))


class MainWindow(QDialog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowFlags(
            QtCore.Qt.WindowTitleHint
            | QtCore.Qt.WindowMinimizeButtonHint
            | QtCore.Qt.WindowCloseButtonHint
            | QtCore.Qt.WindowMaximizeButtonHint
        )

        # 初始化所有必要属性
        self.cam_id = 0
        self.weight_path = ""
        self.processor = None
        self.cap = None
        self.read_cam = False
        self.is_rtsp_stream = False
        self.is_video_file = False
        self.is_detecting = False
        self.input_stream = None
        self.last_time = None

        # 初始化UI
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        # 定时器初始化
        self.timer = QTimer(self)
        self.timer_detect = QTimer(self)
        self.timer.timeout.connect(self.read_stream)
        self.timer_detect.timeout.connect(self.stream_detect)

        # 初始化按钮样式
        self._update_start_button_style(False)
        # 集中连接信号
        self._connect_signals()

    def _connect_signals(self):
        """集中连接 UI 信号到处理方法"""
        self.ui.btn_open_weights.clicked.connect(self.open_weight)
        self.ui.btn_open_image.clicked.connect(self.open_image)
        self.ui.btn_open_video.clicked.connect(self.open_video)
        self.ui.btn_open_rtsp.clicked.connect(self.open_rtsp_stream)
        self.ui.btn_toggle_cam.clicked.connect(self.toggle_camera)
        self.ui.btn_start.clicked.connect(self.start_detect)
        self.ui.video_slider.sliderMoved.connect(self.slider_moved)

    def _get_fps(self, source_type="cam"):
        """Helper to safely retrieve FPS setting from QLineEdit."""
        line_edit = self.ui.cam_fps_input if source_type == "cam" else self.ui.output_fps_input
        default = self.ui.DEFAULT_CAM_FPS if source_type == "cam" else self.ui.DEFAULT_OUTPUT_FPS
        try:
            return int(line_edit.text())
        except ValueError:
            return default

    def _display_cv_image(self, rgb_frame, label: QLabel):
        """Helper to convert an RGB OpenCV frame (numpy array) to QPixmap and display it."""
        if rgb_frame is None:
            return
        h, w, ch = rgb_frame.shape
        qt_img = QImage(rgb_frame.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _show_frame(self, frame, label):
        """封装重复的图像缩放/转换/显示逻辑"""
        if frame is None:
            return
        show = cv2.resize(frame, (640, 480))
        show_rgb = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        self._display_cv_image(show_rgb, label)
        return show  # 返回缩放后的BGR帧

    def _update_start_button_style(self, is_enabled):
        """更新开始按钮的样式"""
        if is_enabled:
            self.ui.btn_start.setStyleSheet("""
                QPushButton {
                    border-radius: 3px;
                    background-color: #4CAF50; 
                    color: white;
                    font-size: 12px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
                QPushButton:pressed {
                    background-color: #3e8e41;
                }
            """)
        else:
            self.ui.btn_start.setStyleSheet("""
                QPushButton {
                    border-radius: 3px;
                    background-color: #cccccc; 
                    color: #666666;
                    font-size: 12px;
                }
            """)

    def slider_moved(self, position):
        """视频进度条拖动"""
        if self.cap and self.is_video_file:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, position)

    def can_run(self):
        """检查运行条件是否满足，并更新开始按钮状态和样式"""
        has_source = self.input_stream is not None or bool(self.ui.image_path.text())
        has_model = bool(self.weight_path) and self.processor is not None
        is_ready = has_source and has_model
        
        self.ui.btn_start.setEnabled(is_ready)
        self._update_start_button_style(is_ready)
        return is_ready

    def is_stream(self):
        """判断当前输入是否为流模式"""
        return self.read_cam or self.is_rtsp_stream or self.is_video_file

    def stop_stream(self, clear_ui=True):
        """停止所有流和定时器，统一初始化状态"""
        # 停止定时器
        self.timer.stop()
        self.timer_detect.stop()

        # 释放资源
        if self.cap:
            self.cap.release()
            self.cap = None

        # 统一初始化状态变量
        self.read_cam = False
        self.is_rtsp_stream = False
        self.is_video_file = False
        self.is_detecting = False
        self.input_stream = None
        self.last_time = None

        # 清理UI
        if clear_ui:
            input_widgets = [self.ui.image_path, self.ui.video_path, self.ui.input_rtsp]
            for widget in input_widgets:
                widget.clear()
            
            self.ui.image_in.setText("原始图像")
            self.ui.image_out.setText("识别结果")
            self.ui.btn_toggle_cam.setText("打开摄像头")
            self.ui.video_slider.setMaximum(0)
            self.ui.video_slider.setValue(0)
            self.ui.video_slider.setEnabled(False)

        self.ui.btn_start.setText("开始运行")
        self.can_run()

    def open_weight(self):
        """加载模型文件"""
        self.ui.btn_open_weights.setText("加载中")
        weights, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "./", "模型文件(*.pt)")
        if weights:
            try:
                self.ui.weight_path.setText(weights)
                self.weight_path = weights
                self.processor = AlgorithmArrangement(weights=self.weight_path)
                # 命令行打印成功信息
                print(f"模型加载成功：{weights}")
                self.ui.btn_open_weights.setText("更改")
            except Exception as e:
                # 命令行打印错误信息
                print(f"模型加载失败：{e}")
                self.processor = None
                self.ui.btn_open_weights.setText("选择模型")
        else:
            self.ui.btn_open_weights.setText("选择模型")

        self.can_run()

    def open_image(self):
        """打开本地图片"""
        self.stop_stream()
        image, _ = QFileDialog.getOpenFileName(self, "选择图像", "./", "图像文件(*.jpg *.png)")
        if image:
            self.input_stream = image
            self.ui.image_path.setText(image)
            # 直接显示图片
            self.ui.image_in.setPixmap(QPixmap(image).scaled(
                self.ui.image_in.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
            print(f"已选择本地图片：{image}")
        self.can_run()

    def open_video(self):
        """打开本地视频"""
        self.stop_stream()
        video, _ = QFileDialog.getOpenFileName(self, "选择视频", "./", "视频文件(*.mp4 *.avi)")
        if video:
            self.cap = cv2.VideoCapture(video)
            if not self.cap.isOpened():
                # 命令行打印错误
                print(f"[错误] 无法打开视频文件：{video}")
                return

            # 设置视频模式
            self.is_video_file = True
            self.ui.video_path.setText(video)
            # 设置进度条
            self.ui.video_slider.setMaximum(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            self.ui.video_slider.setValue(0)

            # 读取并显示第一帧
            _, initial_frame = self.cap.read()
            self.input_stream = self._show_frame(initial_frame, self.ui.image_in)
            print(f"已选择本地视频：{video}")

        self.can_run()

    def open_rtsp_stream(self):
        """打开RTSP流"""
        rtsp_url = self.ui.input_rtsp.text().strip()
        if not rtsp_url:
            # 命令行打印警告
            print(f"请输入有效的 RTSP 地址")
            return

        self.stop_stream()
        self.cap = cv2.VideoCapture(rtsp_url)
        if self.cap.isOpened():
            self.is_rtsp_stream = True
            self.ui.input_rtsp.setText(rtsp_url)

            # 重试读取第一帧
            self.input_stream = None
            for _ in range(3):
                _, frame = self.cap.read()
                if frame is not None:
                    self.input_stream = self._show_frame(frame, self.ui.image_in)
                    break
                time.sleep(0.5)

            if self.input_stream is None:
                print(f"RTSP 流连接成功，但暂时无法读取画面：{rtsp_url}")
            else:
                print(f"已成功连接 RTSP 流：{rtsp_url}")
        else:
            # 命令行打印错误
            print(f"无法连接到 RTSP 地址：{rtsp_url}")
            self.stop_stream()

        self.can_run()

    def toggle_camera(self):
        """切换摄像头状态"""
        if self.read_cam:
            self.stop_stream()
            print(f"已关闭摄像头 (ID: {self.cam_id})")
        else:
            self.stop_stream()
            self.cap = cv2.VideoCapture(self.cam_id)
            ret, frame = self.cap.read()
            if ret:
                self.read_cam = True
                self.input_stream = self._show_frame(frame, self.ui.image_in)
                self.ui.btn_toggle_cam.setText("关闭摄像头")
                print(f"已开启摄像头 (ID: {self.cam_id})")
            else:
                # 命令行打印错误
                print(f"无法打开摄像头 (ID: {self.cam_id})")
                self.stop_stream()

        self.can_run()

    def read_stream(self):
        """读取流数据（摄像头/RTSP/视频）"""
        if self.cap is None:
            return

        # 实时流（摄像头/RTSP）：清空缓冲区读取最新帧
        if self.read_cam or self.is_rtsp_stream:
            last_valid_img = None
            for _ in range(5):
                if self.cap.grab():
                    ret, img = self.cap.retrieve()
                    if ret:
                        last_valid_img = img
            if last_valid_img is not None:
                self.input_stream = self._show_frame(last_valid_img, self.ui.image_in)
            return

        # 视频文件：逐帧读取
        if self.is_video_file:
            success, img = self.cap.read()
            if success:
                self.ui.video_slider.setValue(int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)))
                self.input_stream = self._show_frame(img, self.ui.image_in)
            else:
                # 视频播放结束
                self.timer.stop()
                self.timer_detect.stop()
                self.is_detecting = False
                self.ui.btn_start.setText("播放结束 (重新开始)")
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.ui.video_slider.setValue(0)
                print(f"视频播放结束")
                self.can_run()

    def stream_detect(self):
        """帧检测逻辑"""
        try:
            start_time = time.time()
            source = self.input_stream

            # 处理单张图片模式
            if isinstance(source, str):
                source = cv2.imread(source)
                if not self.is_stream() and self.is_detecting:
                    # 单次图片检测完成后停止检测定时器
                    self.timer_detect.stop()
                    self.is_detecting = False
                    self.ui.btn_start.setText("开始运行")
                    self.can_run()

            if source is None:
                return

            # 使用副本进行处理，防止算法修改原始帧
            frame_to_process = source.copy()

            # 调用算法处理器
            show = self.processor.process(frame_to_process)

            if show is None:
                show = frame_to_process  # 如果算法未返回图像，则显示输入图像

            # BGR 转换成 RGB 用于显示
            show_rgb = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)

            # 计算并绘制 FPS/推理时间
            infer_time = (time.time() - start_time) * 1000
            now = time.time()
            fps = 1.0 / (now - self.last_time) if self.last_time else 0.0
            self.last_time = now
            text = f"{infer_time:.1f} ms | FPS: {fps:.1f}"

            cv2.putText(
                show_rgb,
                text,
                (show_rgb.shape[1] - 200, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 255, 0),
                2,
            )

            self._display_cv_image(show_rgb, self.ui.image_out)

        except Exception as e:
            # 停止检测定时器，防止错误无限循环
            if self.timer_detect.isActive():
                self.timer_detect.stop()
            # 命令行打印检测错误
            print(f"检测过程中发生错误：{e}")
            self.is_detecting = False
            self.ui.btn_start.setText("开始运行")
            # 重新启用数据源按钮
            for btn in [self.ui.btn_toggle_cam, self.ui.btn_open_image, 
                        self.ui.btn_open_video, self.ui.btn_open_rtsp]:
                btn.setEnabled(True)
            self.can_run()

    def start_detect(self):
        """开始/停止检测"""
        if not self.can_run():
            # 命令行打印警告
            print(f"请先加载模型和选择数据源")
            return

        # 流模式（摄像头/RTSP/视频）
        if self.is_stream():
            self.is_detecting = not self.is_detecting
            if self.is_detecting:
                # 启动定时器
                cam_fps = self._get_fps("cam")
                output_fps = self._get_fps("output")
                self.timer.start(int(1000 / cam_fps))
                self.timer_detect.start(int(1000 / output_fps))
                
                # 禁用数据源按钮，启用进度条
                for btn in [self.ui.btn_toggle_cam, self.ui.btn_open_image, 
                            self.ui.btn_open_video, self.ui.btn_open_rtsp]:
                    btn.setEnabled(False)
                self.ui.video_slider.setEnabled(self.is_video_file)
                self.ui.btn_start.setText("停止")
                print(f"开始检测（流模式）- 视频源帧率：{cam_fps} | 输出帧率：{output_fps}")
            else:
                # 停止检测
                self.timer.stop()
                self.timer_detect.stop()
                for btn in [self.ui.btn_toggle_cam, self.ui.btn_open_image, 
                            self.ui.btn_open_video, self.ui.btn_open_rtsp]:
                    btn.setEnabled(True)
                self.ui.video_slider.setEnabled(False)
                self.ui.btn_start.setText("开始运行")
                print(f"停止检测（流模式）")

        # 静态图片模式
        else:
            self.ui.btn_start.setText("运行中...")
            self.ui.btn_start.setEnabled(False)
            self.is_detecting = True
            print(f"开始检测（图片模式）")
            self.stream_detect()
            self.is_detecting = False
            self.ui.btn_start.setText("开始运行")
            self.can_run()
            print(f"图片检测完成")


if __name__ == "__main__":
    # 统一设置 DPI 缩放
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())