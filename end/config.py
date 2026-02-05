# config.py - 端侧设备配置
import os

# ================= 视频源配置 =================

VIDEO_PATH = os.getenv("VIDEO_PATH", r"D:\python_projects\dpsk-test\videos\oil-leak2.mp4")

# ================= 火灾检测配置 =================

FIRE_BRIGHTNESS_THRESHOLD = int(os.getenv("FIRE_BRIGHTNESS_THRESHOLD", "220"))  # 亮度极高点
FIRE_PIXEL_RATIO = float(os.getenv("FIRE_PIXEL_RATIO", "0.005"))               # 火光颜色占比超过0.5%即触发

# ================= 帧差法运动检测配置 =================

MOTION_DIFF_THRESHOLD = int(os.getenv("MOTION_DIFF_THRESHOLD", "50"))      # 提高此值，过滤轻微光影变动 (原为30)
MOTION_AREA_RATIO = float(os.getenv("MOTION_AREA_RATIO", "0.08"))          # 提高此值，要求更大面积的变动 (原为0.05)
GAUSSIAN_BLUR_SIZE = int(os.getenv("GAUSSIAN_BLUR_SIZE", "21"))            # 模糊核大小，必须是奇数

# ================= 定时与采样配置 =================

SYNC_INTERVAL = int(os.getenv("SYNC_INTERVAL", "10"))                      # 定时同步间隔（秒），调试模式设为10秒
POST_TRIGGER_FRAMES = int(os.getenv("POST_TRIGGER_FRAMES", "1"))           # 触发后帧数
FRAME_SKIP = int(os.getenv("FRAME_SKIP", "10"))                            # 帧跳过数量

# ================= 边缘服务器连接配置 =================

EDGE_SERVER_HOST = os.getenv("EDGE_SERVER_HOST", "127.0.0.1")
EDGE_SERVER_PORT = int(os.getenv("EDGE_SERVER_PORT", "8172"))
EDGE_SERVER_URL = os.getenv("EDGE_SERVER_URL", f"http://{EDGE_SERVER_HOST}:{EDGE_SERVER_PORT}")
UPLOAD_TIMEOUT = int(os.getenv("UPLOAD_TIMEOUT", "10"))

# ================= 心跳配置 =================

HEARTBEAT_INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL", "10"))  # 心跳发送间隔（秒）
DEVICE_TIMEOUT = int(os.getenv("DEVICE_TIMEOUT", "120"))         # 设备超时时间（秒）

# ================= Debug配置 =================

DEBUG_MODE = os.getenv("DEBUG_MODE", "true").lower() == "true"                          # 是否启用调试模式
DEBUG_SAVE_FRAMES = os.getenv("DEBUG_SAVE_FRAMES", "false").lower() == "true"           # 是否保存调试帧
DEBUG_OUTPUT_DIR = os.getenv("DEBUG_OUTPUT_DIR", "debug_out")                             # 调试输出目录
DEBUG_PRINT_INTERVAL = int(os.getenv("DEBUG_PRINT_INTERVAL", "10"))                       # 调试信息打印间隔（秒）

# ================= 推理配置 =================

INFERENCE_INTERVAL = float(os.getenv("INFERENCE_INTERVAL", "2.0"))  # 推理间隔（秒），多久检测一次火灾/运动


def print_config():
    """打印端侧设备配置"""
    print("=" * 60)
    print("端侧设备配置")
    print("=" * 60)
    print(f"视频路径: {VIDEO_PATH}")
    print(f"边缘服务器: {EDGE_SERVER_URL}")
    print(f"心跳间隔: {HEARTBEAT_INTERVAL}s")
    print(f"推理间隔: {INFERENCE_INTERVAL}s")
    print(f"火灾检测阈值: 亮度>{FIRE_BRIGHTNESS_THRESHOLD}, 占比>{FIRE_PIXEL_RATIO}")
    print(f"运动检测阈值: 差值>{MOTION_DIFF_THRESHOLD}, 面积比>{MOTION_AREA_RATIO}")
    print(f"调试模式: {DEBUG_MODE}")
    print(f"保存调试帧: {DEBUG_SAVE_FRAMES}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
