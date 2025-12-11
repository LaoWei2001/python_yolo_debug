import detect
import cv2
import numpy as np

# def obj_detect(path):
#     opt = detect.parse_opt()
#     opt.source = source_path
#     detect.run(**vars(opt))

drawing = False
mode = False
ix, iy = -1, -1
point = []
i = 0
# def draw_polygon(event, x, y, flags, param):


def draw_circle(event, x, y, flags, param):
    print(x, y)
    if event == cv2.EVENT_LBUTTONDBLCLK:  # 如果双击鼠标，则画圆
        cv2.circle(
            img,
            center=(x, y),
            radius=100,
            color=(0, 0, 255),
            thickness=1,
        )


def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    # elif event == cv2.EVENT_MOUSEMOVE:
    #     if drawing == True:
    #         cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 1)


def draw_line(event, x, y, flags, param):  # 多条线段
    global point, i
    if event == cv2.EVENT_LBUTTONDOWN:
        point.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)  # 画红点标记
        print(f"标记点{len(point)}:({x},{y})")
        cv2.line(img, point[i - 1], point[i], (0, 255, 0), 2)
        i = i + 1


def draw_quad(event, x, y, flags, param):  # 顺时针或逆时针标记4个点，画一个四边形
    global point, i
    complete_flag = 0
    if event == cv2.EVENT_LBUTTONDOWN:
        point.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)  # 画红点标记


source_path = r"F:\yolov5.7.0_safetyhelmets - 2\yolov5\data\images\ppe_0000_jpg.rf.12dba82119baadab649f2fb1fda2afff.jpg"
img = cv2.imread(source_path)
img_temp = img
cv2.namedWindow("image")
cv2.setMouseCallback("image", draw_quad)  # 将函数与窗口绑定

while 1:
    cv2.imshow("image", img)
    key = cv2.waitKey(20) & 0xFF
    if key == 13:
        for i in range(len(point)):
            cv2.line(img, point[i - 1], point[i], (0, 255, 0), 2)
    elif key == 27:
        break
cv2.destroyAllWindows()
