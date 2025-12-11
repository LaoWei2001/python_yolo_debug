# simple_detect_demo.py
import torch
import cv2
import pandas as pd

def simple_detect_demo():
    """
    简单的检测演示程序，展示如何推理单张图片并获取所有输出信息
    """
    # 1. 加载模型
    model_path = r'F:\lab_project\flame_detection_yolo\yolov5\yolov5m.pt'
    model = torch.hub.load('.', 'custom', path=model_path, source='local', force_reload=True)
    
    # 2. 加载测试图片
    image_path = r'F:\lab_project\flame_detection_yolo\yolov5\data\images\bus.jpg'
    img = cv2.imread(image_path)
    
    if img is None:
        print("无法读取图片，请检查图片路径")
        return
    
    print(f"原始图片尺寸: {img.shape}")
    
    # 3. 执行推理
    print("开始执行推理...")
    results = model(img, size=640)  # 推理图片
    
    # 4. 获取并展示推理结果的各种信息
    
    # 4.1 获取原始检测结果 (pandas DataFrame格式)
    detections = results.pandas().xyxy[0]
    print("\n=== 检测结果 (DataFrame格式) ===")
    print(detections)
    print(f"\n检测到 {len(detections)} 个目标")
    
    # 4.2 遍历每个检测结果
    print("\n=== 逐个目标详细信息 ===")
    # iterrows()是pandas DataFrame对象的一个方法,用于按行遍历
    for index, row in detections.iterrows():
        print(f"目标编号 {index}:")
        print(f"标签名称: {row['name']}")
        print(f"置信度: {row['confidence']:.4f}")
        print(f"边界框坐标:{row['xmin']},{row['ymin']},{row['xmax']},{row['ymax']}")
        print(f"类别索引: {row['class']}")
    
    # 4.3 检查特定类别的目标 (例如火焰)
    print("=== 火焰检测信息 ===")
    flame_detections = detections[detections['name'].str.contains('flame', case=False, na=False)]
    if len(flame_detections) > 0:
        print(f"检测到 {len(flame_detections)} 个火焰目标:")
        for index, row in flame_detections.iterrows():
            print(f"  火焰 {index+1}: 置信度 {row['confidence']:.4f}, 位置 ({row['xmin']}, {row['ymin']}) - ({row['xmax']}, {row['ymax']})")
    else:
        print("未检测到火焰目标")
    
    # 4.4 其他有用信息
    print("\n=== 模型其他输出信息 ===")
    print(f"Results对象类型: {type(results)}")
    print(f"检测结果数量: {len(results.xyxy)}")  # batch中的图片数量
    if len(results.xyxy) > 0:
        print(f"第一张图片的检测张量形状: {results.xyxy[0].shape}")  # [N, 6] N个检测框，每个6个值(x1,y1,x2,y2,conf,class)
    
    # 4.5 渲染带检测框的图片
    results.render()  # 在图像上绘制检测框
    rendered_img = results.ims[0]  # 获取渲染后的图像
    
    # 4.6 保存结果图片
    output_path = "detection_result.jpg"
    cv2.imwrite(output_path, rendered_img)
    print(f"\n结果图片已保存到: {output_path}")
    
    # 4.7 显示图片 (可选)
    cv2.imshow('Detection Result', rendered_img)
    print("按任意键关闭图片窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    simple_detect_demo()