"""
边缘服务器 - 云端部署版本
支持从环境变量配置，便于在 47.108.54.154 云端部署
"""
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
from dataclasses import dataclass, field
from threading import Lock

# 导入配置
try:
    from config import (
        EDGE_SERVER_HOST, EDGE_SERVER_PORT, EDGE_ID,
        CLOUD_API_URL, CLOUD_HEARTBEAT_URL, CLOUD_BASE_URL,
        YOLO_MODEL_PATH, CONF_THRESHOLD, ENABLE_SAEC_ADAPTIVE,
        SC_THRESHOLD, WEIGHTS, DEVICE_TIMEOUT,
        ENABLE_AUTO_MODEL_UPDATE, MODEL_CHECK_INTERVAL,
        HARD_EXAMPLE_THRESHOLD, HARD_EXAMPLE_CACHE_DIR,
        CLOUD_MODEL_CHECK_URL, CLOUD_HARD_EXAMPLE_URL, CLOUD_DISTILL_URL,
        print_config
    )
except ImportError:
    # 如果config.py不存在，使用默认配置
    print("⚠️ 未找到config.py，使用默认配置")
    EDGE_SERVER_HOST = "0.0.0.0"
    EDGE_SERVER_PORT = 8172
    EDGE_ID = "EDGE-001"
    CLOUD_SERVER_HOST = "127.0.0.1"
    CLOUD_SERVER_PORT = 9876
    CLOUD_BASE_URL = f"http://{CLOUD_SERVER_HOST}:{CLOUD_SERVER_PORT}"
    CLOUD_API_URL = f"{CLOUD_BASE_URL}/api/v1/inspect"
    CLOUD_HEARTBEAT_URL = f"{CLOUD_BASE_URL}/api/v1/edge_heartbeat"
    YOLO_MODEL_PATH = "/app/models/best.pt"
    CONF_THRESHOLD = 0.45
    ENABLE_SAEC_ADAPTIVE = True
    SC_THRESHOLD = 6.36
    WEIGHTS = {"entropy": 0.25, "edge": 0.20, "sharpness": 0.20, "gradient": 0.20, "jpeg_residual": 0.15}
    DEVICE_TIMEOUT = 120
    ENABLE_AUTO_MODEL_UPDATE = False
    MODEL_CHECK_INTERVAL = 3600
    HARD_EXAMPLE_THRESHOLD = 0.6
    HARD_EXAMPLE_CACHE_DIR = "./hard_examples"

# 尝试导入YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ YOLO未安装，将使用模拟模式")

app = FastAPI(title=f"Edge Server {EDGE_ID}")

# 加载边缘模型
edge_model = None
if YOLO_AVAILABLE and os.path.exists(YOLO_MODEL_PATH):
    edge_model = YOLO(YOLO_MODEL_PATH)
    print(f"✅ 边缘模型加载成功: {YOLO_MODEL_PATH}")
elif YOLO_AVAILABLE:
    print(f"⚠️ 警告: 未找到边缘模型 {YOLO_MODEL_PATH}")
else:
    print("⚠️ YOLO不可用，运行在模拟模式")


# ================= 设备管理 =================

@dataclass
class DeviceInfo:
    """设备信息"""
    device_id: str
    first_seen: float
    last_heartbeat: float
    status: str = "normal"
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
    """设备管理器"""

    def __init__(self, timeout_seconds: int = 120):
        self.devices: Dict[str, DeviceInfo] = {}
        self.timeout_seconds = timeout_seconds
        self._lock = Lock()
        self._edge_online = True

    def update_device(self, device_id: str, status: str = "normal",
                      error_message: str = "", video_source: str = "",
                      inference_interval: float = 2.0):
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
            return device

    def increment_uploads(self, device_id: str):
        with self._lock:
            if device_id in self.devices:
                self.devices[device_id].total_uploads += 1

    def get_device(self, device_id: str) -> Optional[DeviceInfo]:
        with self._lock:
            return self.devices.get(device_id)

    def get_all_devices(self) -> Dict[str, DeviceInfo]:
        with self._lock:
            return dict(self.devices)

    def check_offline_devices(self) -> List[str]:
        now = time.time()
        offline_devices = []
        with self._lock:
            for device_id, device in self.devices.items():
                if device.status != "offline" and (now - device.last_heartbeat) > self.timeout_seconds:
                    device.status = "offline"
                    offline_devices.append(device_id)
        return offline_devices

    def get_statistics(self) -> dict:
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


device_manager = DeviceManager(timeout_seconds=DEVICE_TIMEOUT)


# ================= SAEC评估器 =================

class SAEC_Estimator:
    """场景复杂度评估器"""

    @staticmethod
    def compute_grayscale_entropy(gray):
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.ravel() / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-7))
        return min(entropy / 8.0, 1.0)

    @staticmethod
    def compute_edge_density(gray):
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges / 255.0) / (gray.shape[0] * gray.shape[1])
        return min(edge_density / 0.06, 1.0)

    @staticmethod
    def compute_laplacian_variance(gray):
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return 1.0 - min(laplacian_var / 600.0, 1.0)

    @staticmethod
    def compute_gradient_magnitude(gray):
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        mean_gradient = np.mean(gradient_magnitude)
        return min(mean_gradient / 50.0, 1.0)

    @staticmethod
    def compute_jpeg_residual(image):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        _, encoded = cv2.imencode('.jpg', image, encode_param)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        residual = cv2.absdiff(image, decoded)
        residual_gray = cv2.cvtColor(residual, cv2.COLOR_BGR2GRAY)
        residual_score = np.mean(residual_gray) / 255.0
        return min(residual_score * 10, 1.0)

    @classmethod
    def get_sc_score(cls, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        entropy = cls.compute_grayscale_entropy(gray)
        edge = cls.compute_edge_density(gray)
        sharpness = cls.compute_laplacian_variance(gray)
        gradient = cls.compute_gradient_magnitude(gray)
        jpeg_residual = cls.compute_jpeg_residual(image)

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
                    yolo_detections: int, reason: str, sc_score: float = 0.0):
    """上报到云端服务器"""
    try:
        _, img_encoded = cv2.imencode('.jpg', image_np)
        img_bytes = img_encoded.tobytes()

        files = {'image': ('image.jpg', img_bytes, 'image/jpeg')}
        data = {
            'device_id': device_id,
            'edge_id': EDGE_ID,
            'trigger_reason': reason,
            'sc_score': str(sc_score)
        }

        response = requests.post(CLOUD_API_URL, files=files, data=data, timeout=30)
        if response.status_code == 200:
            result = response.json()
            print(f"☁️ [云端响应] 级别: {result.get('alert_level', 'N/A')}")
            return result
        else:
            print(f"❌ [云端上报失败] HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ [云端上报异常] {e}")
        return None


def send_edge_heartbeat():
    """边缘服务器向云端发送心跳"""
    try:
        stats = device_manager.get_statistics()
        payload = {
            'edge_id': EDGE_ID,
            'timestamp': time.time(),
            'device_stats': stats,
            'status': 'online'
        }
        response = requests.post(CLOUD_HEARTBEAT_URL, json=payload, timeout=5)
        return response.status_code == 200
    except Exception as e:
        print(f"⚠️ 边缘到云端心跳失败: {e}")
        return False


# ================= 协同训练扩展 (预留) =================

def check_model_update():
    """检查云端是否有新模型版本"""
    if not ENABLE_AUTO_MODEL_UPDATE:
        return
    try:
        response = requests.get(CLOUD_MODEL_CHECK_URL, timeout=5)
        if response.status_code == 200:
            data = response.json()
            latest_version = data.get("model_version")
            # TODO: 与本地版本对比，如有新版本则下载
            print(f"📦 最新模型版本: {latest_version}")
    except Exception as e:
        print(f"⚠️ 检查模型更新失败: {e}")


def upload_hard_example(image_np: np.ndarray, yolo_result: dict, device_id: str, sc_score: float):
    """上传难例到云端用于增量学习"""
    try:
        # 只上传低置信度或高复杂度的样本
        max_conf = max([box.get("conf", 0) for box in yolo_result.get("boxes", [])], default=0)
        if max_conf > HARD_EXAMPLE_THRESHOLD and sc_score < SC_THRESHOLD:
            return  # 不是难例，不上传

        _, img_encoded = cv2.imencode('.jpg', image_np)
        img_b64 = base64.b64encode(img_encoded).decode('utf-8')

        payload = {
            "device_id": device_id,
            "edge_id": EDGE_ID,
            "image_base64": img_b64,
            "yolo_prediction": yolo_result,
            "sc_score": sc_score,
            "timestamp": time.time(),
            "reason": "hard_example"
        }

        response = requests.post(CLOUD_HARD_EXAMPLE_URL, json=payload, timeout=10)
        if response.status_code == 200:
            print(f"📦 难例已上传 | 设备:{device_id}")
    except Exception as e:
        print(f"⚠️ 难例上传失败: {e}")


def request_distillation_labels(image_np: np.ndarray, device_id: str) -> Optional[Dict]:
    """请求云端大模型的软标签用于知识蒸馏"""
    try:
        _, img_encoded = cv2.imencode('.jpg', image_np)
        img_b64 = base64.b64encode(img_encoded).decode('utf-8')

        payload = {
            "device_id": device_id,
            "edge_id": EDGE_ID,
            "image_base64": img_b64,
            "temperature": 2.0
        }

        response = requests.post(CLOUD_DISTILL_URL, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"⚠️ 蒸馏请求失败: {e}")
    return None


# ================= API请求模型 =================

class HeartbeatRequest(BaseModel):
    device_id: str
    timestamp: float
    status: str = "normal"
    error_message: str = ""
    video_source: str = ""
    inference_interval: float = 2.0


class TrainingStatusResponse(BaseModel):
    """训练状态响应"""
    edge_id: str
    current_model_version: str
    pending_updates: bool
    last_training_time: Optional[float]
    hard_examples_cached: int


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
    """主推理接口"""
    if device_id != "unknown":
        device_manager.increment_uploads(device_id)

    results_manifest = []

    for file in files:
        data = await file.read()
        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            continue

        # SAEC场景复杂度评估
        sc_score, sc_details = SAEC_Estimator.get_sc_score(img)
        is_complex = sc_score > SC_THRESHOLD

        # YOLO检测
        yolo_detections = 0
        yolo_boxes = []
        if edge_model:
            yolo_res = edge_model.predict(img, conf=CONF_THRESHOLD, verbose=False)[0]
            yolo_detections = len(yolo_res.boxes)
            yolo_boxes = [
                {"cls": int(box.cls), "conf": float(box.conf), "xyxy": box.xyxy.tolist()}
                for box in yolo_res.boxes
            ]

        # 协同调度决策
        decision = "Edge_Only"
        is_fire = "FIRE" in reason.upper()

        if is_fire:
            print(f"🔥 [火灾告警] 设备:{device_id}")
            decision = "Fire_Alert_Direct"
            background_tasks.add_task(
                report_to_cloud,
                img.copy(),
                device_id,
                yolo_detections,
                reason,
                sc_score
            )
        elif ENABLE_SAEC_ADAPTIVE:
            if is_complex or (yolo_detections == 0 and sc_score > 0.4):
                print(f"🚀 [协同调度] 设备:{device_id} Sc={sc_score}")
                decision = "Edge_Cloud_Collaborative"
                background_tasks.add_task(
                    report_to_cloud,
                    img.copy(),
                    device_id,
                    yolo_detections,
                    reason,
                    sc_score
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
        "model_path": YOLO_MODEL_PATH,
        "cloud_url": CLOUD_BASE_URL,
        "device_stats": device_manager.get_statistics(),
        "timestamp": time.time()
    }


# ================= 协同训练扩展接口 (预留) =================

@app.get("/training/status", response_model=TrainingStatusResponse)
async def get_training_status():
    """
    获取边缘训练状态
    云端可通过此接口了解边缘模型情况
    """
    # TODO: 实现真实状态查询
    hard_example_count = 0
    if os.path.exists(HARD_EXAMPLE_CACHE_DIR):
        hard_example_count = len([f for f in os.listdir(HARD_EXAMPLE_CACHE_DIR) if f.endswith('.jpg')])

    return TrainingStatusResponse(
        edge_id=EDGE_ID,
        current_model_version="yolov10n-v1.0.0",
        pending_updates=False,
        last_training_time=None,
        hard_examples_cached=hard_example_count
    )


@app.post("/training/model/update")
async def trigger_model_update():
    """
    触发手动模型更新检查
    正常情况自动检查，此接口用于强制触发
    """
    check_model_update()
    return {"status": "checking", "edge_id": EDGE_ID}


@app.post("/training/local-cache/clear")
async def clear_hard_example_cache():
    """清空本地难例缓存"""
    cleared = 0
    if os.path.exists(HARD_EXAMPLE_CACHE_DIR):
        for f in os.listdir(HARD_EXAMPLE_CACHE_DIR):
            if f.endswith('.jpg') or f.endswith('.json'):
                os.remove(os.path.join(HARD_EXAMPLE_CACHE_DIR, f))
                cleared += 1
    return {"status": "cleared", "files_removed": cleared}


# ================= 后台任务 =================

async def periodic_tasks():
    """周期性后台任务"""
    model_check_counter = 0
    while True:
        # 检查离线设备
        offline_devices = device_manager.check_offline_devices()
        if offline_devices:
            print(f"📴 检测到 {len(offline_devices)} 个设备离线")

        # 向云端发送心跳
        send_edge_heartbeat()

        # 定期检查模型更新
        model_check_counter += 30
        if model_check_counter >= MODEL_CHECK_INTERVAL:
            check_model_update()
            model_check_counter = 0

        await asyncio.sleep(30)


@app.on_event("startup")
async def startup_event():
    """启动事件"""
    print_config()
    # 创建难例缓存目录
    os.makedirs(HARD_EXAMPLE_CACHE_DIR, exist_ok=True)
    asyncio.create_task(periodic_tasks())
    print(f"🚀 边缘服务器启动 | ID: {EDGE_ID} | 端口: {EDGE_SERVER_PORT}")


if __name__ == "__main__":
    uvicorn.run(app, host=EDGE_SERVER_HOST, port=EDGE_SERVER_PORT)
