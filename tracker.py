import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

class KalmanTracker:
    """
    对应 C++ 中的 KalmanTracker 类
    使用 OpenCV 的 KalmanFilter 实现
    """
    count = 0

    def __init__(self, init_rect):
        # 状态向量: [cx, cy, s, r, dx, dy, ds]
        # 观测向量: [cx, cy, s, r]
        self.kf = cv2.KalmanFilter(7, 4, 0)
        
        # 转移矩阵 (Transition Matrix)
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ], np.float32)

        # 测量矩阵, 噪声等参数设置
        self.kf.measurementMatrix = np.eye(4, 7, dtype=np.float32)
        self.kf.processNoiseCov = np.eye(7, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(7, dtype=np.float32)

        # 初始化状态
        # init_rect 格式: [x1, y1, x2, y2]
        x1, y1, x2, y2 = init_rect
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2
        s = w * h # area
        r = w / (h + 1e-6)

        # statePost 必须是 float32
        self.kf.statePost = np.array([[cx], [cy], [s], [r], [0], [0], [0]], np.float32)
        
        self.time_since_update = 0
        self.id = KalmanTracker.count
        KalmanTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def predict(self):
        """
        预测下一帧位置
        """
        # 避免面积变为负数
        if((self.kf.statePost[6, 0] + self.kf.statePost[2, 0]) <= 0):
            self.kf.statePost[6, 0] *= 0.0

        self.kf.predict()
        self.age += 1
        
        if(self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        
        # [修复] 使用 [i, 0] 获取标量值，而不是数组
        s = self.kf.statePost
        return self.get_rect_xysr(s[0, 0], s[1, 0], s[2, 0], s[3, 0])

    def update(self, state_mat):
        """
        使用观测值更新状态
        state_mat: [x1, y1, x2, y2]
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1

        x1, y1, x2, y2 = state_mat
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2
        s = w * h
        r = w / (h + 1e-6)

        measurement = np.array([[cx], [cy], [s], [r]], np.float32)
        self.kf.correct(measurement)

    def get_state(self):
        """
        获取当前估计的 [x1, y1, x2, y2]
        """
        s = self.kf.statePost
        # [修复] 使用 [i, 0] 获取标量值
        return self.get_rect_xysr(s[0, 0], s[1, 0], s[2, 0], s[3, 0])

    def get_rect_xysr(self, cx, cy, s, r):
        w = np.sqrt(s * r)
        h = s / w
        x = (cx - w / 2)
        y = (cy - h / 2)
        
        # 简单的边界处理
        if x < 0 and cx > 0: x = 0
        if y < 0 and cy > 0: y = 0
            
        return [x, y, x + w, y + h]


class SortTracker:
    def __init__(self, max_age=20, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets):
        """
        输入: [[x1, y1, x2, y2, score], ...]
        输出: [[x1, y1, x2, y2, track_id], ...]
        """
        self.frame_count += 1
        
        # 1. 预测现有轨迹
        trks = np.zeros((len(self.trackers), 5)) 
        to_del = []
        
        for t, trk in enumerate(self.trackers):
            pos = trk.predict() # 这里的 pos 现在是纯数字列表了
            # 这里的赋值操作之前会报错，因为 pos 里包含了 array
            trks[t, :] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
                
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        # 2. 匹配
        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(dets, trks)

        # 3. 更新匹配的轨迹
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                self.trackers[t].update(dets[d, :4][0])

        # 4. 创建新轨迹
        for i in unmatched_dets:
            trk = KalmanTracker(dets[i, :4])
            self.trackers.append(trk)

        # 5. 输出结果
        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

    def associate_detections_to_trackers(self, detections, trackers):
        # [修复] 如果没有跟踪器，unmatched_trackers 应该返回空的 1D 数组，而不是 2D
        if(len(trackers)==0):
            return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0),dtype=int)

        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = self.iou(det, trk)

        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > self.iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                row_ind, col_ind = linear_sum_assignment(-iou_matrix)
                matched_indices = np.stack((row_ind, col_ind), axis=1)
        else:
            matched_indices = np.empty((0,2))

        unmatched_detections = []
        for d, det in enumerate(detections):
            if(d not in matched_indices[:,0]):
                unmatched_detections.append(d)
        
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if(t not in matched_indices[:,1]):
                unmatched_trackers.append(t)

        matches = []
        for m in matched_indices:
            if(iou_matrix[m[0], m[1]] < self.iou_threshold):
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1,2))
        
        if(len(matches)==0):
            matches = np.empty((0,2),dtype=int)
        else:
            matches = np.concatenate(matches,axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

    def iou(self, bb_test, bb_gt):
        xx1 = np.maximum(bb_test[0], bb_gt[0])
        yy1 = np.maximum(bb_test[1], bb_gt[1])
        xx2 = np.minimum(bb_test[2], bb_gt[2])
        yy2 = np.minimum(bb_test[3], bb_gt[3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1]) + 
                  (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
        return(o)