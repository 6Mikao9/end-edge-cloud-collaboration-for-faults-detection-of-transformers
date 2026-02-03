import os
import cv2
import time
import json
import asyncio
import hashlib
import requests
import numpy as np
import uvicorn
from datetime import datetime, timedelta
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Set
from pydantic import BaseModel
from ultralytics import YOLO
from dataclasses import dataclass, field
from threading import Lock

# ================= 配置区 =================
YOLO_MODEL_PATH = r"D:\python_projects\dpsk-test\edge\runs\detect\transformer_project\yolov10_optimized5\weights\best.pt"

# 云端API配置
# 本地开发测试: 127.0.0.1:9876
# 云端部署: 47.108.54.154:9876
import os
CLOUD_SERVER_HOST = os.getenv("CLOUD_SERVER_HOST", "127.0.0.1")  # 默认本地
CLOUD_SERVER_PORT = int(os.getenv("CLOUD_SERVER_PORT", "9876"))
CLOUD_BASE_URL = f"http://{CLOUD_SERVER_HOST}:{CLOUD_SERVER_PORT}"

CLOUD_API_URL = f"{CLOUD_BASE_URL}/api/v1/inspect"
CLOUD_HEARTBEAT_URL = f"{CLOUD_BASE_URL}/api/v1/edge_heartbeat"
EDGE_ID = os.getenv("EDGE_ID", "EDGE-001")  # 边缘服务器ID，用于标记设备

# SAEC配置
ENABLE_SAEC_ADAPTIVE = True
SC_THRESHOLD = 6.36
WEIGHTS = {
    "entropy": 0.25,      # 灰度级熵
    "edge": 0.20,         # 边缘密度
    "sharpness": 0.20,    # 拉普拉斯方差
    "gradient": 0.20,     # 梯度幅值
    "jpeg_residual": 0.15 # JPEG残差
}

CONF_THRESHOLD = 0.45
DEVICE_TIMEOUT = 120  # 设备超时时间（秒）

# 加载边缘模型
edge_model = None
if os.path.exists(YOLO_MODEL_PATH):
    edge_model = YOLO(YOLO_MODEL_PATH)
    print(f"✅ 边缘模型加载成功: {YOLO_MODEL_PATH}")
else:
    print(f"⚠️ 警告: 未找到边缘模型 {YOLO_MODEL_PATH}")


# ================= 设备管理 =================

@dataclass
class DeviceInfo:
    """设备信息"""
    device_id: str
    first_seen: float
    last_heartbeat: float
    status: str = "normal"  # normal, warning, error, offline
    error_message: str = ""
    video_source: str = ""
    inference_interval: float = 2.0
    total_uploads: int = 0
    
    def to_dict(self):
        return {
            "device_id": self.device_id,
            "first_seen": datetime.fromtimestamp(self.first_seen).isoformat(),
            "last_heartbeat": datetime.fromtimestamp(self.last_heartbeat).isoformat(),
            "status": self.status,
            "error_message": self.error_message,
            "video_source": self.video_source,
            "inference_interval": self.inference_interval,
            "total_uploads": self.total_uploads
        }


class DeviceManager:
    """设备管理器 - 管理所有连接的端侧设备"""
    
    def __init__(self, timeout_seconds: int = 120):
        self.devices: Dict[str, DeviceInfo] = {}
        self.timeout_seconds = timeout_seconds
        self._lock = Lock()
        self._edge_online = True
    
    def update_device(self, device_id: str, status: str = "normal", 
                      error_message: str = "", video_source: str = "",
                      inference_interval: float = 2.0):
        """更新设备状态"""
        with self._lock:
            now = time.time()
            if device_id not in self.devices:
                self.devices[device_id] = DeviceInfo(
                    device_id=device_id,
                    first_seen=now,
                    last_heartbeat=now,
                    video_source=video_source,
                    inference_interval=inference_interval
                )
                print(f"🆕 新设备上线: {device_id}")
            
            device = self.devices[device_id]
            device.last_heartbeat = now
            device.status = status
            device.error_message = error_message
            
            if status == "offline":
                print(f"📴 设备离线: {device_id}")
            elif status in ["warning", "error"]:
                print(f"⚠️ 设备异常 [{status}]: {device_id} - {error_message}")
            
            return device
    
    def increment_uploads(self, device_id: str):
        """增加设备上传计数"""
        with self._lock:
            if device_id in self.devices:
                self.devices[device_id].total_uploads += 1
    
    def get_device(self, device_id: str) -> Optional[DeviceInfo]:
        """获取设备信息"""
        with self._lock:
            return self.devices.get(device_id)
    
    def get_all_devices(self) -> Dict[str, DeviceInfo]:
        """获取所有设备"""
        with self._lock:
            return dict(self.devices)
    
    def check_offline_devices(self) -> List[str]:
        """检查并标记离线设备"""
        now = time.time()
        offline_devices = []
        with self._lock:
            for device_id, device in self.devices.items():
                if device.status != "offline" and (now - device.last_heartbeat) > self.timeout_seconds:
                    device.status = "offline"
                    offline_devices.append(device_id)
                    print(f"📴 设备标记为离线: {device_id}")
        return offline_devices
    
    def get_statistics(self) -> dict:
        """获取设备统计信息"""
        with self._lock:
            total = len(self.devices)
            normal = sum(1 for d in self.devices.values() if d.status == "normal")
            warning = sum(1 for d in self.devices.values() if d.status == "warning")
            error = sum(1 for d in self.devices.values() if d.status == "error")
            offline = sum(1 for d in self.devices.values() if d.status == "offline")
            
            return {
                "total_devices": total,
                "normal": normal,
                "warning": warning,
                "error": error,
                "offline": offline,
                "online_rate": round((normal + warning) / total * 100, 2) if total > 0 else 0
            }


# 全局设备管理器
device_manager = DeviceManager(timeout_seconds=DEVICE_TIMEOUT)


# ================= 增强版SAEC评估器 =================

class SAEC_Estimator:
    """
    增强版场景复杂度评估器
    实现参考架构中的5个指标：
    1. 灰度级熵 (Grayscale Entropy)
    2. 边缘密度 (Edge Density)  
    3. 拉普拉斯方差 (Laplacian Variance)
    4. 梯度幅值 (Gradient Magnitude)
    5. JPEG残差 (JPEG Residual)
    """

    @staticmethod
    def compute_grayscale_entropy(gray):
        """计算灰度级熵"""
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.ravel() / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-7))
        return min(entropy / 8.0, 1.0)  # 归一化到0-1

    @staticmethod
    def compute_edge_density(gray):
        """计算边缘密度"""
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges / 255.0) / (gray.shape[0] * gray.shape[1])
        return min(edge_density / 0.06, 1.0)  # 0.06为经验阈值

    @staticmethod
    def compute_laplacian_variance(gray):
        """计算拉普拉斯方差（清晰度）"""
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # 方差小表示模糊，复杂度得分高
        return 1.0 - min(laplacian_var / 600.0, 1.0)

    @staticmethod
    def compute_gradient_magnitude(gray):
        """计算梯度幅值"""
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        mean_gradient = np.mean(gradient_magnitude)
        return min(mean_gradient / 50.0, 1.0)  # 归一化

    @staticmethod
    def compute_jpeg_residual(image):
        """计算JPEG残差（压缩伪影检测）"""
        # 模拟JPEG压缩再解码
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        _, encoded = cv2.imencode('.jpg', image, encode_param)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        
        # 计算残差
        residual = cv2.absdiff(image, decoded)
        residual_gray = cv2.cvtColor(residual, cv2.COLOR_BGR2GRAY)
        residual_score = np.mean(residual_gray) / 255.0
        return min(residual_score * 10, 1.0)  # 放大并归一化

    @classmethod
    def get_sc_score(cls, image):
        """计算综合场景复杂度得分"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 计算5个指标
        entropy = cls.compute_grayscale_entropy(gray)
        edge = cls.compute_edge_density(gray)
        sharpness = cls.compute_laplacian_variance(gray)
        gradient = cls.compute_gradient_magnitude(gray)
        jpeg_residual = cls.compute_jpeg_residual(image)
        
        # 加权求和
        sc_score = (
            WEIGHTS["entropy"] * entropy +
            WEIGHTS["edge"] * edge +
            WEIGHTS["sharpness"] * sharpness +
            WEIGHTS["gradient"] * gradient +
            WEIGHTS["jpeg_residual"] * jpeg_residual
        )
        
        return round(float(sc_score), 3), {
            "entropy": round(entropy, 3),
            "edge": round(edge, 3),
            "sharpness": round(sharpness, 3),
            "gradient": round(gradient, 3),
            "jpeg_residual": round(jpeg_residual, 3)
        }


# ================= 云端通信 =================

def report_to_cloud(image_np: np.ndarray, device_id: str, 
                    yolo_detections: int, reason: str, skip_llm: bool = False,
                    yolo_boxes: list = None):
    """上报到云端服务器，包含YOLO检测框"""
    try:
        _, img_encoded = cv2.imencode('.jpg', image_np)
        img_bytes = img_encoded.tobytes()
        
        files = {'image': ('image.jpg', img_bytes, 'image/jpeg')}
        data = {
            'device_id': device_id,
            'edge_id': EDGE_ID,
            'trigger_reason': reason,
            'skip_llm': str(skip_llm),
            'yolo_boxes': json.dumps(yolo_boxes or [])  # YOLO检测框
        }
        
        response = requests.post(CLOUD_API_URL, files=files, data=data, timeout=30)
        if response.status_code == 200:
            result = response.json()
            print(f"☁️ [云端响应] 记录ID: {result.get('record_id', 'N/A')}, 级别: {result.get('alert_level', 'N/A')}")
            return result
        else:
            print(f"❌ [云端上报失败] HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ [云端上报异常] {e}")
        return None


def send_edge_heartbeat():
    """边缘服务器向云端发送心跳，包含完整设备列表"""
    try:
        stats = device_manager.get_statistics()
        devices = device_manager.get_all_devices()

        # 构建完整设备列表
        device_list = []
        for device_id, device in devices.items():
            device_list.append({
                'device_id': device_id,
                'status': device.status,
                'last_heartbeat': device.last_heartbeat,
                'total_uploads': device.total_uploads
            })

        payload = {
            'edge_id': EDGE_ID,
            'timestamp': time.time(),
            'device_stats': stats,
            'devices': device_list,  # 新增：完整设备列表
            'status': 'online'
        }
        response = requests.post(CLOUD_HEARTBEAT_URL, json=payload, timeout=5)
        if response.status_code == 200:
            print(f"💓 心跳发送成功 | 设备数:{len(device_list)} 在线率:{stats.get('online_rate', 0)}%")
        return response.status_code == 200
    except Exception as e:
        print(f"⚠️ 边缘到云端心跳失败: {e}")
        return False


# ================= FastAPI应用创建 (必须在路由之前) =================
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理 - 替代@on_event(消除弃用警告)"""
    print(f"🚀 边缘服务器启动 | ID: {EDGE_ID}")
    print(f"☁️ 云端服务器: {CLOUD_BASE_URL}")
    asyncio.create_task(periodic_tasks())
    yield
    print("🛑 边缘服务器关闭")


# 创建FastAPI实例 - 使用lifespan
app = FastAPI(title="Edge Server with Device Management & SAEC", lifespan=lifespan)

# ================= API请求模型 =================

class HeartbeatRequest(BaseModel):
    device_id: str
    timestamp: float
    status: str = "normal"
    error_message: str = ""
    video_source: str = ""
    inference_interval: float = 2.0


# ================= API接口 =================

@app.post("/heartbeat")
async def device_heartbeat(request: HeartbeatRequest):
    """端侧设备心跳接口"""
    device = device_manager.update_device(
        device_id=request.device_id,
        status=request.status,
        error_message=request.error_message,
        video_source=request.video_source,
        inference_interval=request.inference_interval
    )
    
    return {
        "status": "success",
        "edge_time": time.time(),
        "device_status": device.status,
        "message": "Heartbeat received"
    }


@app.post("/predict")
async def predict(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    reason: str = Form("unknown"),
    timestamp: float = Form(0),
    device_id: str = Form("unknown")
):
    """主推理接口 - 接收端侧上传的图像"""
    
    # 更新设备上传计数
    if device_id != "unknown":
        device_manager.increment_uploads(device_id)
    
    results_manifest = []
    
    for file in files:
        # 读取图像
        data = await file.read()
        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            continue
        
        # 1. SAEC场景复杂度评估
        sc_score, sc_details = SAEC_Estimator.get_sc_score(img)
        is_complex = sc_score > SC_THRESHOLD
        
        # 2. 边缘YOLO检测 - 获取检测框用于云端对比显示
        yolo_detections = 0
        yolo_boxes = []
        if edge_model:
            yolo_res = edge_model.predict(img, conf=CONF_THRESHOLD, verbose=False)[0]
            yolo_detections = len(yolo_res.boxes)
            img_h, img_w = img.shape[:2]
            for box in yolo_res.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                # 转换为归一化坐标 (0-1000)，与Qwen3-VL格式一致 [y1, x1, y2, x2]
                yolo_boxes.append({
                    "cls": int(box.cls),
                    "conf": float(box.conf),
                    "box_norm": [
                        int(y1 / img_h * 1000),  # y1
                        int(x1 / img_w * 1000),  # x1
                        int(y2 / img_h * 1000),  # y2
                        int(x2 / img_w * 1000)   # x2
                    ],
                    "box_px": [int(x1), int(y1), int(x2), int(y2)]  # 像素坐标
                })
        
        # 3. 协同调度决策
        decision = "Edge_Only"
        
        # 火灾直接上报，不经过复杂度判断
        is_fire = "FIRE" in reason.upper()
        
        if is_fire:
            print(f"🔥 [火灾告警] 设备:{device_id} 检测到火灾，立即上报云端！")
            decision = "Fire_Alert_Direct"
            background_tasks.add_task(
                report_to_cloud, 
                img.copy(), 
                device_id, 
                yolo_detections,
                reason,
                skip_llm=False,
                yolo_boxes=yolo_boxes
            )
        elif ENABLE_SAEC_ADAPTIVE:
            # 场景复杂或YOLO无信心时触发云端
            if is_complex or (yolo_detections == 0 and sc_score > 0.4):
                print(f"🚀 [协同调度] 设备:{device_id} 场景复杂(Sc={sc_score}), 触发云端MLLM")
                decision = "Edge_Cloud_Collaborative"
                # 异步上报云端，携带YOLO检测框
                background_tasks.add_task(
                    report_to_cloud, 
                    img.copy(), 
                    device_id, 
                    yolo_detections,
                    reason,
                    skip_llm=False,
                    yolo_boxes=yolo_boxes
                )
        
        results_manifest.append({
            "filename": file.filename,
            "sc_score": sc_score,
            "is_complex": is_complex,
            "yolo_count": yolo_detections,
            "decision": decision,
            "sc_details": sc_details
        })
    
    return {
        "status": "success",
        "edge_id": EDGE_ID,
        "device_id": device_id,
        "saec_enabled": ENABLE_SAEC_ADAPTIVE,
        "results": results_manifest
    }


@app.get("/devices")
async def get_devices():
    """获取所有设备列表"""
    devices = device_manager.get_all_devices()
    return {
        "status": "success",
        "edge_id": EDGE_ID,
        "devices": {k: v.to_dict() for k, v in devices.items()},
        "statistics": device_manager.get_statistics()
    }


@app.get("/devices/{device_id}")
async def get_device_detail(device_id: str):
    """获取单个设备详情"""
    device = device_manager.get_device(device_id)
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    return {
        "status": "success",
        "device": device.to_dict()
    }


@app.get("/health")
async def health_check():
    """边缘服务器健康检查"""
    return {
        "status": "healthy",
        "edge_id": EDGE_ID,
        "model_loaded": edge_model is not None,
        "device_stats": device_manager.get_statistics(),
        "timestamp": time.time()
    }


# ================= 后台任务 =================

async def periodic_tasks():
    """周期性后台任务"""
    while True:
        # 检查离线设备
        offline_devices = device_manager.check_offline_devices()
        if offline_devices:
            print(f"📴 检测到 {len(offline_devices)} 个设备离线")

        # 向云端发送心跳
        send_edge_heartbeat()

        await asyncio.sleep(30)  # 每30秒执行一次


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8172)
