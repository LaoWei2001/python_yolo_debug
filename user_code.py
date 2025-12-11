import cv2
import torch
import numpy as np
from models.common import DetectMultiBackend
from utils.general import (
    check_img_size,
    scale_boxes,
    non_max_suppression,
    Profile,
)
from utils.torch_utils import select_device
from utils.plots import Annotator, colors
from utils.augmentations import letterbox

# 导入跟踪器
from tracker import SortTracker

#########  全局变量定义区域  #########
cout = 0
# 前后两帧
frame_1 = 0
frame_2 = 0


#####################################
class AlgorithmArrangement:
    def __init__(
        self,
        weights,
        device="",
        imgsz=(640, 640),
        conf_thres=0.25,
        iou_thres=0.5,
        max_det=1000,
        line_thickness=2,
        font_size=0.5,
        hide_labels=False,
        hide_conf=False,
    ):
        # 1. 初始化参数
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.line_thickness = line_thickness
        self.font_size = font_size
        self.hide_labels = hide_labels
        self.hide_conf = hide_conf

        # [新增] 初始化 SORT 跟踪器
        self.tracker = SortTracker(max_age=20, min_hits=3, iou_threshold=0.3)

        # 2. 加载模型
        self.device = select_device(device)
        self.model = DetectMultiBackend(
            weights, device=self.device, dnn=False, data=None, fp16=False
        )
        self.stride, self.names, self.pt = (
            self.model.stride,
            self.model.names,
            self.model.pt,
        )
        self.imgsz = check_img_size(imgsz, s=self.stride)

        # 3. 模型热身
        self.model.warmup(imgsz=(1, 3, *self.imgsz))

        # 4. 计时工具
        self.dt = (Profile(), Profile())

        # 原始的绘制函数 (保留作为备份，但主要使用 draw_tracks)
        # self.draw_detections

    # 专门用于绘制跟踪结果的函数
    def draw_tracks(self, im0, tracks):
        annotator = Annotator(
            im0,
            line_width=self.line_thickness,
            font_size=self.font_size,
            example=str(self.names),
        )

        # tracks 格式: [[x1, y1, x2, y2, track_id], ...]
        for x1, y1, x2, y2, track_id in tracks:
            # 构造标签，显示 ID
            label = f"ID: {int(track_id)}"

            # 使用 track_id 作为颜色索引，确保同一个 ID 颜色保持一致
            color = colors(int(track_id), True)

            # 画框和标签
            annotator.box_label([x1, y1, x2, y2], label, color=color)

        return annotator.result()

    def preprocess_frame(self, frame):
        # 1. letterbox 保持比例，并填充至 640×640
        img = letterbox(frame, self.imgsz, stride=self.stride, auto=False)[0]

        # 2. BGR → RGB，HWC → CHW
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        # 3. 像素归一化
        img = torch.from_numpy(img).to(self.device).float() / 255.0

        # 4. 变成4维的tensor
        if img.ndim == 3:
            img = img.unsqueeze(0)

        return img

    def detect(self, img_tensor):
        # 推理
        with self.dt[0]:
            pred = self.model(img_tensor, augment=False, visualize=False)

        # NMS
        with self.dt[1]:
            pred = non_max_suppression(
                pred,
                self.conf_thres,
                self.iou_thres,
                classes=None,
                agnostic=False,
                max_det=self.max_det,
            )
        return pred

    def draw_detections(self, im0, det):
        annotator = Annotator(
            im0,
            line_width=self.line_thickness,
            font_size=self.font_size,
            example=str(self.names),
        )
        # 遍历tensor获取信息
        for *xyxy, conf, cls in det:
            c = int(cls)
            label = (
                None
                if self.hide_labels
                else (
                    self.names[c] if self.hide_conf else f"{self.names[c]} {conf:.2f}"
                )
            )
            # 画框和标签
            annotator.box_label(xyxy, label, color=colors(c, True))

        return annotator.result()

    def draw(self, frame, pred, img_preprocessed):
        im0 = frame.copy()

        for det in pred:
            if len(det):
                # 将 det 里的框映射回原图 im0
                det[:, :4] = scale_boxes(
                    img_preprocessed.shape[2:],  # ← 模型输入尺寸
                    det[:, :4],
                    im0.shape,  # ← 原图尺寸
                ).round()

                im0 = self.draw_detections(im0, det)

        return im0

    # (保留原有函数: detect_draw - 如果 GUI 框架只调用 process，这个函数可能不再被使用)
    def detect_draw(self, frame):
        img_tensor = self.preprocess_frame(frame)
        pred = self.detect(img_tensor)
        frame_out = self.draw(frame, pred, img_tensor)
        return frame_out

    # [修改/替代] 核心处理函数，用于 GUI 循环
    def process(self, frame):
        """
        核心函数，执行检测和跟踪，返回带有 ID 标注的视频帧。
        """
        global cout
        cout += 1

        # 1. 预处理,将图片转为tensor
        img_tensor = self.preprocess_frame(frame)

        # 2. 推理 (YOLO Detect)
        pred = self.detect(img_tensor)

        # 3. 数据转换: Tensor -> Numpy for Tracker
        dets_to_sort = np.empty((0, 5))

        # 处理 batch 中的第一张图 (通常是一帧)
        det = pred[0]

        if len(det):
            # 将坐标映射回原图尺寸
            det[:, :4] = scale_boxes(
                img_tensor.shape[2:], det[:, :4], frame.shape
            ).round()

            # 提取 [x1, y1, x2, y2, conf] 用于跟踪
            # 注意：det 是 Tensor，需要转为 numpy
            # det[:, :5] 包含了 [x1, y1, x2, y2, conf]
            dets_to_sort = det[:, :5].cpu().numpy()

        # 4. 更新跟踪器
        # 输入: [[x1, y1, x2, y2, score], ...]
        # 输出: [[x1, y1, x2, y2, track_id], ...]
        track_result = self.tracker.update(dets_to_sort)

        # 5. 绘制跟踪结果
        frame_out = frame.copy()

        if len(track_result) > 0:
            frame_out = self.draw_tracks(frame_out, track_result)

        return frame_out
