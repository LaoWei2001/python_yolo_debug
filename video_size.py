import cv2
import os

def get_video_resolution(video_path):
    """
    获取视频文件的分辨率（宽×高）
    :param video_path: 视频文件的绝对/相对路径
    :return: 元组 (width, height)，失败则返回 None
    """
    # 检查文件是否存在
    if not os.path.exists(video_path):
        print(f"❌ 错误：文件 '{video_path}' 不存在！")
        return None
    
    # 检查是否是有效文件（而非文件夹）
    if not os.path.isfile(video_path):
        print(f"❌ 错误：'{video_path}' 不是有效文件！")
        return None
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 错误：无法打开视频文件（可能格式不支持/文件损坏/权限不足）！")
        return None
    
    # 获取分辨率（宽、高）
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 释放资源
    cap.release()
    
    return (width, height)

# ===================== 核心修改：直接指定视频路径 =====================
# 替换成你的视频文件路径（注意路径写法）
# 示例1：相对路径（视频和脚本在同一文件夹）
video_path = r"F:\video\魏\01000001081000000.mp4"  

# 示例2：Windows绝对路径（推荐用r前缀避免转义）
# video_path = r"C:\Users\你的用户名\Videos\电影片段.mp4"

# 示例3：Linux/Mac绝对路径
# video_path = "/home/你的用户名/Videos/clip.mkv"
# ====================================================================

# 执行获取并输出结果
if __name__ == "__main__":
    resolution = get_video_resolution(video_path)
    if resolution:
        width, height = resolution
        print(f"✅ 视频路径：{video_path}")
        print(f"✅ 视频分辨率：{width} × {height}")
    else:
        print("❌ 获取分辨率失败，请检查路径或文件是否有效！")