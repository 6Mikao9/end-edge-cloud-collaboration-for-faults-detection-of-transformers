# config.py - 端侧设备配置

VIDEO_PATH = "D:\\python_projects\\dpsk-test\\videos\\fire2.mp4"

# 1. 颜色/起火检测优化
FIRE_BRIGHTNESS_THRESHOLD = 220  # 亮度极高点
FIRE_PIXEL_RATIO = 0.005         # 火光颜色占比超过0.5%即触发

# 2. 帧差法灵敏度优化
MOTION_DIFF_THRESHOLD = 50       # 提高此值，过滤轻微光影变动 (原为30)
MOTION_AREA_RATIO = 0.08         # 提高此值，要求更大面积的变动 (原为0.05)
GAUSSIAN_BLUR_SIZE = 21          # 模糊核大小，必须是奇数

# 3. 定时与采样
SYNC_INTERVAL = 60
POST_TRIGGER_FRAMES = 1
FRAME_SKIP = 10

# 4. 边缘服务器连接配置
EDGE_SERVER_URL = "http://127.0.0.1:8172"
UPLOAD_TIMEOUT = 10

# 5. 心跳配置
HEARTBEAT_INTERVAL = 30  # 心跳发送间隔（秒）
DEVICE_TIMEOUT = 120     # 设备超时时间（秒）
