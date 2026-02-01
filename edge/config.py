"""
边缘服务器配置文件
支持从环境变量读取，便于云端部署
"""
import os

# ================= 服务器配置 =================

# 边缘服务器自身配置
EDGE_SERVER_HOST = os.getenv("EDGE_SERVER_HOST", "0.0.0.0")
EDGE_SERVER_PORT = int(os.getenv("EDGE_SERVER_PORT", "8172"))
EDGE_ID = os.getenv("EDGE_ID", "EDGE-001")

# ================= 云端服务器配置 =================
# 部署到云端时，将以下配置改为云服务器的公网IP

# 云端服务器地址 (默认本地测试，云端部署时修改)
CLOUD_SERVER_HOST = os.getenv("CLOUD_SERVER_HOST", "127.0.0.1")
CLOUD_SERVER_PORT = int(os.getenv("CLOUD_SERVER_PORT", "9876"))

# 构建完整URL
CLOUD_BASE_URL = f"http://{CLOUD_SERVER_HOST}:{CLOUD_SERVER_PORT}"
CLOUD_API_URL = f"{CLOUD_BASE_URL}/api/v1/inspect"
CLOUD_HEARTBEAT_URL = f"{CLOUD_BASE_URL}/api/v1/edge_heartbeat"

# 协同训练接口 (预留)
CLOUD_TRAINING_BASE = f"{CLOUD_BASE_URL}/api/v1/training"
CLOUD_MODEL_CHECK_URL = f"{CLOUD_TRAINING_BASE}/model/latest"
CLOUD_HARD_EXAMPLE_URL = f"{CLOUD_TRAINING_BASE}/hard-example"
CLOUD_DISTILL_URL = f"{CLOUD_TRAINING_BASE}/distill"

# ================= YOLO模型配置 =================

# 模型路径 (云端部署时需确保路径正确)
YOLO_MODEL_PATH = os.getenv(
    "YOLO_MODEL_PATH",
    r"/app/models/best.pt"  # Linux云端默认路径
    if os.name != 'nt' else
    r"D:\python_projects\dpsk-test\edge\runs\detect\transformer_project\yolov10_test6\weights\best.pt"
)

# 推理置信度阈值
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.45"))

# ================= SAEC复杂度评估配置 =================

ENABLE_SAEC_ADAPTIVE = os.getenv("ENABLE_SAEC_ADAPTIVE", "true").lower() == "true"
SC_THRESHOLD = float(os.getenv("SC_THRESHOLD", "6.36"))

# SAEC权重配置
WEIGHTS = {
    "entropy": float(os.getenv("SAEC_WEIGHT_ENTROPY", "0.25")),
    "edge": float(os.getenv("SAEC_WEIGHT_EDGE", "0.20")),
    "sharpness": float(os.getenv("SAEC_WEIGHT_SHARPNESS", "0.20")),
    "gradient": float(os.getenv("SAEC_WEIGHT_GRADIENT", "0.20")),
    "jpeg_residual": float(os.getenv("SAEC_WEIGHT_JPEG", "0.15"))
}

# ================= 设备管理配置 =================

DEVICE_TIMEOUT = int(os.getenv("DEVICE_TIMEOUT", "120"))  # 设备超时时间（秒）

# ================= 协同训练配置 (预留) =================

# 是否启用自动模型更新
ENABLE_AUTO_MODEL_UPDATE = os.getenv("ENABLE_AUTO_MODEL_UPDATE", "false").lower() == "true"

# 模型检查间隔（秒）
MODEL_CHECK_INTERVAL = int(os.getenv("MODEL_CHECK_INTERVAL", "3600"))

# 难例上传阈值 (置信度低于此值视为难例)
HARD_EXAMPLE_THRESHOLD = float(os.getenv("HARD_EXAMPLE_THRESHOLD", "0.6"))

# 本地难例缓存目录
HARD_EXAMPLE_CACHE_DIR = os.getenv("HARD_EXAMPLE_CACHE_DIR", "./hard_examples")

# ================= 日志配置 =================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def print_config():
    """打印当前配置（调试用）"""
    print("=" * 50)
    print("边缘服务器配置")
    print("=" * 50)
    print(f"边缘ID: {EDGE_ID}")
    print(f"边缘服务: {EDGE_SERVER_HOST}:{EDGE_SERVER_PORT}")
    print(f"云端地址: {CLOUD_BASE_URL}")
    print(f"模型路径: {YOLO_MODEL_PATH}")
    print(f"SAEC启用: {ENABLE_SAEC_ADAPTIVE}")
    print(f"自动更新: {ENABLE_AUTO_MODEL_UPDATE}")
    print("=" * 50)


if __name__ == "__main__":
    print_config()
