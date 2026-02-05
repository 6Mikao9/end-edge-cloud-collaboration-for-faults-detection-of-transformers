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
    r"D:\python_projects\dpsk-test\edge\runs\detect\transformer_rgb\v2_data_boosted4\weights\best.pt"
)

# 推理置信度阈值
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.05"))

# ================= SAEC复杂度评估配置 =================

ENABLE_SAEC_ADAPTIVE = os.getenv("ENABLE_SAEC_ADAPTIVE", "true").lower() == "true"
SC_THRESHOLD = float(os.getenv("SC_THRESHOLD", "0.7"))
SAEC_COMPLEXITY_TRIGGER = float(os.getenv("SAEC_COMPLEXITY_TRIGGER", "0.4"))  # 无检测且复杂度超过此值触发云端

# SAEC权重配置
WEIGHTS = {
    "entropy": float(os.getenv("SAEC_WEIGHT_ENTROPY", "0.25")),      # 灰度级熵
    "edge": float(os.getenv("SAEC_WEIGHT_EDGE", "0.20")),            # 边缘密度
    "sharpness": float(os.getenv("SAEC_WEIGHT_SHARPNESS", "0.20")),  # 拉普拉斯方差
    "gradient": float(os.getenv("SAEC_WEIGHT_GRADIENT", "0.20")),    # 梯度幅值
    "jpeg_residual": float(os.getenv("SAEC_WEIGHT_JPEG", "0.15"))    # JPEG残差
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

# ================= Debug配置 =================

# 是否启用详细调试输出
DEBUG_INFERENCE = os.getenv("DEBUG_INFERENCE", "true").lower() == "true"

# 是否保存调试图像
DEBUG_SAVE_IMAGES = os.getenv("DEBUG_SAVE_IMAGES", "false").lower() == "true"

# 调试图像保存目录
DEBUG_OUTPUT_DIR = os.getenv("DEBUG_OUTPUT_DIR", "./debug_output")

# ================= 日志配置 =================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def print_config():
    """打印当前配置（调试用）"""
    print("=" * 60)
    print("边缘服务器配置")
    print("=" * 60)
    print(f"边缘ID: {EDGE_ID}")
    print(f"边缘服务: {EDGE_SERVER_HOST}:{EDGE_SERVER_PORT}")
    print(f"云端地址: {CLOUD_BASE_URL}")
    print(f"模型路径: {YOLO_MODEL_PATH}")
    print(f"置信度阈值: {CONF_THRESHOLD}")
    print(f"SAEC启用: {ENABLE_SAEC_ADAPTIVE}")
    print(f"SAEC阈值: {SC_THRESHOLD}")
    print(f"自动更新: {ENABLE_AUTO_MODEL_UPDATE}")
    print(f"调试模式: {DEBUG_INFERENCE}")
    print(f"保存调试图像: {DEBUG_SAVE_IMAGES}")
    print("=" * 60)


def debug_print_inference(filename: str, sc_score: float, yolo_count: int, 
                          yolo_boxes: list, decision: str, device_id: str = "unknown"):
    """
    打印推理调试信息到控制台
    
    Args:
        filename: 图像文件名
        sc_score: SAEC场景复杂度分数
        yolo_count: YOLO检测到的目标数量
        yolo_boxes: YOLO检测框详情
        decision: 调度决策
        device_id: 设备ID
    """
    if not DEBUG_INFERENCE:
        return
    
    print("\n" + "=" * 60)
    print(f"🔍 [推理调试] 设备:{device_id} 文件:{filename}")
    print("-" * 60)
    print(f"   SAEC复杂度: {sc_score:.3f} (阈值: {SC_THRESHOLD})")
    print(f"   YOLO检测数: {yolo_count}")
    
    if yolo_boxes:
        print(f"   YOLO检测框详情:")
        for i, box in enumerate(yolo_boxes):
            cls = box.get("cls", "N/A")
            conf = box.get("conf", 0)
            box_px = box.get("box_px", [0, 0, 0, 0])
            print(f"      [{i+1}] 类别:{cls} 置信度:{conf:.2%} 坐标:{box_px}")
    else:
        print(f"   YOLO检测框: 无")
    
    print(f"   调度决策: {decision}")
    print("=" * 60 + "\n")


def save_debug_image(img, yolo_boxes, filename_prefix="debug"):
    """
    保存调试图像（带YOLO检测框）
    
    Args:
        img: OpenCV图像
        yolo_boxes: YOLO检测框
        filename_prefix: 文件名前缀
    """
    if not DEBUG_SAVE_IMAGES:
        return
    
    import cv2
    import time
    
    os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)
    
    # 复制图像用于绘制
    debug_img = img.copy()
    
    # 绘制YOLO检测框
    for i, box in enumerate(yolo_boxes):
        box_px = box.get("box_px", [0, 0, 0, 0])
        x1, y1, x2, y2 = box_px
        conf = box.get("conf", 0)
        cls = box.get("cls", 0)
        
        # 绘制框
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 绘制标签
        label = f"{cls}:{conf:.2f}"
        cv2.putText(debug_img, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 保存图像
    timestamp = int(time.time())
    filename = f"{filename_prefix}_{timestamp}.jpg"
    filepath = os.path.join(DEBUG_OUTPUT_DIR, filename)
    cv2.imwrite(filepath, debug_img)
    print(f"💾 调试图像已保存: {filepath}")


if __name__ == "__main__":
    print_config()
