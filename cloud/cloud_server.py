"""
云端服务器 - 支持 Qwen3-VL 视觉定位 (Grounding)
- 接收边缘服务器消息
- 调用 Swift Deploy Qwen3-VL 接口进行视觉定位
- 前端 Canvas 绘制检测框
"""

import os
import cv2
import time
import json
import base64
import asyncio
import uvicorn
import aiohttp
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from threading import Lock

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ================= 日志与配置 =================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SWIFT_DEPLOY_URL = "http://47.108.54.154:7860/v1/chat/completions"
SWIFT_HEALTH_URL = "http://47.108.54.154:7860/health"
SWIFT_MODEL_NAME = "models"

CLOUD_PORT = 9876
TEMP_DIR = "static"
os.makedirs(TEMP_DIR, exist_ok=True)

swift_online = False

# ================= 数据模型 =================

@dataclass
class Device:
    device_id: str
    edge_id: str
    first_seen: float
    last_seen: float
    status: str = "online"  # online/offline/normal/warning/danger
    total_inspections: int = 0
    last_alert: Optional[Dict] = None
    last_alert_time: Optional[float] = None  # 最后报警时间戳


@dataclass
class EdgeServer:
    """边缘服务器状态"""
    edge_id: str
    first_seen: float
    last_heartbeat: float
    status: str = "online"  # online/offline
    device_count: int = 0
    ip_address: Optional[str] = None

@dataclass
class InspectionRecord:
    record_id: str
    timestamp: float
    device_id: str
    edge_id: str
    image_name: str
    trigger_reason: str
    llm_result: Optional[str] = None
    alert_level: str = "normal"
    boxes: List[List[int]] = field(default_factory=list)  # Qwen3-VL检测框 [y1, x1, y2, x2]
    yolo_boxes: List[Dict] = field(default_factory=list)  # YOLO检测框，包含cls/conf/box_norm

# ================= 状态管理 =================

class StateManager:
    def __init__(self):
        self.devices: Dict[str, Device] = {}
        self.edges: Dict[str, EdgeServer] = {}  # 边缘服务器状态
        self.records: List[InspectionRecord] = []
        self.max_records = 100
        self._lock = Lock()
        self.device_timeout = 60  # 设备超时时间（秒），超过则标记为离线
        self.edge_timeout = 120   # 边缘服务器超时时间（秒）

    def update_device(self, device_id: str, edge_id: str, status: str = "online"):
        with self._lock:
            key = f"{edge_id}:{device_id}"
            now = time.time()
            if key not in self.devices:
                self.devices[key] = Device(device_id=device_id, edge_id=edge_id, first_seen=now, last_seen=now)
            self.devices[key].last_seen = now
            # 只更新为更严重的状态，或从offline恢复
            current_status = self.devices[key].status
            if status == "online" and current_status == "offline":
                self.devices[key].status = "online"
            elif status in ["warning", "danger"]:
                self.devices[key].status = status

    def update_edge_heartbeat(self, edge_id: str, device_stats: dict, ip_address: str = None):
        """更新边缘服务器心跳"""
        with self._lock:
            now = time.time()
            if edge_id not in self.edges:
                self.edges[edge_id] = EdgeServer(
                    edge_id=edge_id,
                    first_seen=now,
                    last_heartbeat=now,
                    ip_address=ip_address
                )
                logger.info(f"🆕 新边缘服务器上线: {edge_id}")
            else:
                self.edges[edge_id].last_heartbeat = now
                self.edges[edge_id].status = "online"
            self.edges[edge_id].device_count = device_stats.get("total_devices", 0)

    def check_edge_and_device_timeout(self):
        """检查边缘服务器和设备超时状态"""
        with self._lock:
            now = time.time()
            # 检查边缘服务器超时
            for edge_id, edge in self.edges.items():
                if edge.status == "online" and (now - edge.last_heartbeat) > self.edge_timeout:
                    edge.status = "offline"
                    logger.warning(f"📴 边缘服务器离线: {edge_id}")

            # 检查设备超时
            for key, device in self.devices.items():
                if device.status != "offline" and (now - device.last_seen) > self.device_timeout:
                    device.status = "offline"
                    logger.info(f"📴 设备离线: {key}")

    def add_record(self, record: InspectionRecord):
        with self._lock:
            self.records.insert(0, record)
            if len(self.records) > self.max_records:
                self.records.pop()
            key = f"{record.edge_id}:{record.device_id}"
            if key in self.devices:
                self.devices[key].total_inspections += 1
                self.devices[key].last_alert = {
                    "level": record.alert_level,
                    "time": record.timestamp,
                    "reason": record.trigger_reason
                }
                # 只有非normal级别的报警才更新last_alert_time
                if record.alert_level != "normal":
                    self.devices[key].last_alert_time = record.timestamp
                    self.devices[key].status = record.alert_level  # warning/danger

    def check_device_status_timeout(self, timeout_seconds: int = 300):
        """检查设备是否超过timeout_seconds无报警，是则置为normal"""
        with self._lock:
            now = time.time()
            for key, device in self.devices.items():
                # 只处理当前处于warning/danger状态的设备
                if device.status in ["warning", "danger"]:
                    if device.last_alert_time is not None:
                        elapsed = now - device.last_alert_time
                        if elapsed > timeout_seconds:
                            device.status = "normal"
                            logger.info(f"✅ 设备 {key} 超过{timeout_seconds//60}分钟无新报警，自动置为正常状态")
                    else:
                        # 有报警记录但没有last_alert_time（兼容旧数据），设为当前时间
                        device.last_alert_time = now

    def get_records_json(self):
        with self._lock:
            return [{
                "record_id": r.record_id,
                "timestamp": datetime.fromtimestamp(r.timestamp).isoformat(),
                "device_id": r.device_id,
                "edge_id": r.edge_id,
                "image_url": f"/static/{r.image_name}",
                "trigger_reason": r.trigger_reason,
                "llm_result": r.llm_result,
                "alert_level": r.alert_level,
                "boxes": r.boxes,  # Qwen3-VL检测框
                "yolo_boxes": r.yolo_boxes  # YOLO检测框
            } for r in self.records]

    def get_statistics(self):
        with self._lock:
            return {
                "total_devices": len(self.devices),
                "online_devices": sum(1 for d in self.devices.values() if d.status != "offline"),
                "total_edges": len(self.edges),
                "online_edges": sum(1 for e in self.edges.values() if e.status == "online"),
                "total_inspections": len(self.records),
                "danger_alerts": sum(1 for r in self.records if r.alert_level == "danger")
            }
    
    def get_devices(self):
        with self._lock:
            return list(self.devices.values())

state = StateManager()

# ================= 核心工具：Qwen3-VL 坐标解析 =================

def extract_qwen_boxes(text: str) -> List[List[int]]:
    """解析 Qwen3-VL 视觉定位坐标（已修正正则表达式）"""
    if not text:
        return []
    # ✅ 修正：匹配两个坐标点 (x1,y1),(x2,y2)
    pattern = r"<box>\s*\((\d+),(\d+)\),\s*\((\d+),(\d+)\)\s*</box>"
    matches = re.findall(pattern, text)
    # 转换为整数列表 [[x1, y1, x2, y2], ...]
    boxes = [[int(c) for c in m] for m in matches]
    return boxes


def normalize_box_to_pixel(box: List[int], img_width: int, img_height: int) -> Tuple[int, int, int, int]:
    """
    将归一化坐标 (0-1000) 转换为像素坐标
    
    Args:
        box: [y1, x1, y2, x2] 归一化坐标
        img_width: 图像宽度
        img_height: 图像高度
    
    Returns:
        (x, y, w, h) 像素坐标，适合 Canvas 绘制
    """
    y1, x1, y2, x2 = box
    
    # 转换为像素坐标
    px = int((x1 / 1000) * img_width)
    py = int((y1 / 1000) * img_height)
    pw = int(((x2 - x1) / 1000) * img_width)
    ph = int(((y2 - y1) / 1000) * img_height)
    
    return px, py, pw, ph


async def call_swift_deploy(image_path: str, prompt: str, temperature: float = 0.3) -> str:
    """调用 Swift Deploy Qwen3-VL 进行视觉分析"""
    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        
        payload = {
            "model": SWIFT_MODEL_NAME,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    {"type": "text", "text": prompt}
                ]
            }],
            "temperature": temperature
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(SWIFT_DEPLOY_URL, json=payload, timeout=60) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"]
                return f"Error: HTTP {resp.status}"
    except Exception as e:
        return f"Error: {str(e)}"


def build_grounding_prompt(trigger_reason: str) -> str:
    """构建带视觉定位指令的 Prompt（已修正坐标格式）"""
    #base_instruction = """【设备视觉分析】请结合电力运行规程，对该图像中的设施状态进行安全合规性检查，定位潜在的风险隐患。任务要求：1. 检查是否存在如漏油(最常见）等异常、设备损坏或安全隐患但是请在遵循不放过问题的基础上尽量避免误报，即不要把好的设备也报为故障设备2. 如果发现问题，请详细描述异常位置和特征3. **重要：用坐标框出异常区域** 坐标格式说明：- 使用 <box>(x1,y1),(x2,y2)</box> 标记位置- 坐标范围 0-1000，x 在前 y 在后- 例如：变压器漏油点 <box>(150,300),(400,450)</box>请用中文回复，包含：1. 整体状态评估2. 具体问题描述（如有）3. 异常位置的坐标标记"""
    base_instruction = """【设备视觉分析】请结合电力运行规程，对该图像中的设施状态进行安全合规性检查，定位潜在的风险隐患"""
    if "FIRE" in trigger_reason.upper():
        return f"{base_instruction}\n\n【重点关注】检测到火灾告警，请仔细检查明火、烟雾、过热区域，并用坐标框出所有可疑位置。"
    else:
        return f"{base_instruction}\n\n【触发原因】{trigger_reason}"

def parse_alert_level(trigger_reason: str, result: str, has_boxes: bool) -> str:
    """解析告警级别"""
    # 火灾直接 danger
    if "FIRE" in trigger_reason.upper():
        return "danger"
    
    # 有检测框表示发现异常
    if has_boxes:
        return "warning"
    
    # 检测异常关键词
    abnormal_keywords = ["异常", "缺陷", "破损", "风险", "故障", "损坏", "漏油", "锈蚀"]
    result_lower = result.lower()
    for kw in abnormal_keywords:
        if kw in result_lower:
            return "warning"
    
    return "normal"

# ================= FastAPI应用创建 (必须在路由之前) =================
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理 - 替代@on_event(消除弃用警告)"""
    # 启动时执行
    asyncio.create_task(check_swift_health())
    asyncio.create_task(check_device_timeout())
    asyncio.create_task(check_edge_and_device_connection())  # 新增：检查边缘连接
    logger.info(f"☁️ 云端视觉巡检中心启动 | 端口:{CLOUD_PORT}")
    yield
    # 关闭时执行（预留）
    logger.info("☁️ 云端视觉巡检中心关闭")


# 创建FastAPI实例 - 使用lifespan
app = FastAPI(title="Cloud Vision Inspection Center", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=TEMP_DIR), name="static")

# ================= API 接口 =================

@app.post("/api/v1/inspect")
async def inspect(
    image: UploadFile = File(...),
    device_id: str = Form(...),
    edge_id: str = Form(...),
    trigger_reason: str = Form("unknown"),
    yolo_boxes: str = Form("[]")  # JSON字符串，YOLO检测框
):
    """接收边缘检测请求，进行AI视觉定位分析"""
    logger.info(f"📥 收到检测 | 设备:{device_id} | 原因:{trigger_reason}")
    
    # Motion 不显示
    is_motion_only = "MOTION" in trigger_reason.upper() and "FIRE" not in trigger_reason.upper()
    
    # 保存图像
    timestamp = int(time.time())
    filename = f"{timestamp}_{device_id}.jpg"
    save_path = os.path.join(TEMP_DIR, filename)
    with open(save_path, "wb") as f:
        f.write(await image.read())
    
    state.update_device(device_id, edge_id)

    # 解析YOLO检测框
    try:
        yolo_boxes_parsed = json.loads(yolo_boxes)
    except:
        yolo_boxes_parsed = []
    logger.info(f"📦 收到 {len(yolo_boxes_parsed)} 个YOLO检测框")

    # AI 分析
    if swift_online:
        prompt = build_grounding_prompt(trigger_reason)
        llm_raw = await call_swift_deploy(save_path, prompt)
    else:
        llm_raw = "AI算力平台离线，请人工复核"

    # 解析Qwen3-VL坐标
    boxes = extract_qwen_boxes(llm_raw)
    logger.info(f"🔍 Qwen3-VL检测到 {len(boxes)} 个定位框")

    # 分级
    alert_level = parse_alert_level(trigger_reason, llm_raw, len(boxes) > 0)

    # 记录（Motion 不加入）
    if not is_motion_only:
        record = InspectionRecord(
            record_id=f"REC-{timestamp}",
            timestamp=time.time(),
            device_id=device_id,
            edge_id=edge_id,
            image_name=filename,
            trigger_reason=trigger_reason,
            llm_result=llm_raw,
            alert_level=alert_level,
            boxes=boxes,
            yolo_boxes=yolo_boxes_parsed
        )
        state.add_record(record)
    
    return {
        "status": "success",
        "alert_level": alert_level,
        "boxes_count": len(boxes),
        "llm_result": llm_raw[:200] + "..." if len(llm_raw) > 200 else llm_raw
    }


class EdgeHeartbeatRequest(BaseModel):
    """边缘心跳请求"""
    edge_id: str
    timestamp: float
    device_stats: dict
    devices: Optional[List[dict]] = None  # 完整设备列表（可选）
    status: str = "online"


@app.post("/api/v1/edge_heartbeat")
async def heartbeat(request: EdgeHeartbeatRequest):
    """
    接收边缘服务器心跳
    包含设备统计信息和可选的完整设备列表
    """
    # 更新边缘服务器状态
    state.update_edge_heartbeat(
        edge_id=request.edge_id,
        device_stats=request.device_stats
    )

    # 如果提供了完整设备列表，批量更新设备状态
    if request.devices:
        for device in request.devices:
            state.update_device(
                device_id=device.get("device_id"),
                edge_id=request.edge_id,
                status=device.get("status", "online")
            )
    else:
        # 兼容旧版：只更新单个设备（从device_stats推断）
        total = request.device_stats.get("total_devices", 0)
        if total > 0:
            # 至少更新一个设备以保持活跃
            state.update_device(
                device_id=f"{request.edge_id}-device",
                edge_id=request.edge_id,
                status="online"
            )

    return {"swift_online": swift_online, "status": "ok"}


@app.get("/api/v1/records")
async def get_records():
    return {"records": state.get_records_json()}


@app.get("/api/v1/statistics")
async def get_stats():
    return {"data": state.get_statistics()}


@app.get("/api/v1/devices")
async def get_devices():
    """获取所有设备和边缘服务器状态"""
    with state._lock:
        devices = [
            {**d.__dict__, "first_seen": datetime.fromtimestamp(d.first_seen).isoformat(),
             "last_seen": datetime.fromtimestamp(d.last_seen).isoformat()}
            for d in state.devices.values()
        ]
        edges = [
            {**e.__dict__, "first_seen": datetime.fromtimestamp(e.first_seen).isoformat(),
             "last_heartbeat": datetime.fromtimestamp(e.last_heartbeat).isoformat()}
            for e in state.edges.values()
        ]
    return {
        "devices": devices,
        "edges": edges,
        "summary": {
            "total_edges": len(edges),
            "online_edges": sum(1 for e in edges if e.get("status") == "online"),
            "total_devices": len(devices),
            "online_devices": sum(1 for d in devices if d.get("status") != "offline")
        }
    }


@app.get("/api/v1/health")
async def health():
    return {"swift_online": swift_online, "status": "healthy"}


# ================= 协同训练扩展接口 (Collaborative Training API) =================
# 以下接口为未来大小模型协同训练预留，当前仅返回占位数据

class ModelInfo(BaseModel):
    """模型信息"""
    model_config = {"protected_namespaces": ()}  # 解决字段名冲突警告
    model_version: str
    model_url: Optional[str] = None
    model_hash: Optional[str] = None
    release_notes: Optional[str] = None
    created_at: Optional[float] = None


class HardExampleUpload(BaseModel):
    """难例上传请求"""
    device_id: str
    edge_id: str
    image_base64: str  # Base64编码图像
    yolo_prediction: Dict  # YOLO预测结果（框+置信度）
    sc_score: float  # 场景复杂度分数
    timestamp: float
    reason: str  # 触发原因


class DistillationRequest(BaseModel):
    """知识蒸馏请求"""
    device_id: str
    edge_id: str
    image_base64: str
    temperature: float = 2.0  # 蒸馏温度


class DistillationResponse(BaseModel):
    """知识蒸馏响应 - 软标签"""
    soft_labels: List[float]  # 各类别软概率
    boxes: List[Dict]  # 边界框坐标
    feature_map: Optional[str] = None  # Base64编码的特征图（可选）
    teacher_confidence: float


class TrainingTask(BaseModel):
    """训练任务"""
    task_id: str
    task_type: str  # "distillation", "incremental", "finetune"
    status: str  # "pending", "running", "completed", "failed"
    progress: float  # 0-100
    model_version: str
    created_at: float
    updated_at: float


# ----- 模型管理接口 -----

@app.get("/api/v1/training/model/latest", response_model=ModelInfo)
async def get_latest_model():
    """
    获取最新边缘模型信息
    边缘服务器定期查询，有新版本则下载更新
    """
    # TODO: 实现模型版本管理
    return ModelInfo(
        model_version="yolov10n-v1.0.0",
        model_url=None,  # 预留: "http://47.108.54.154:9000/models/yolov10n-v1.0.0.pt"
        model_hash=None,  # 预留: SHA256哈希校验
        release_notes="初始版本",
        created_at=time.time()
    )


@app.get("/api/v1/training/model/download/{version}")
async def download_model(version: str):
    """
    下载指定版本的模型文件
    用于边缘服务器增量更新
    """
    # TODO: 实现模型文件流式传输
    raise HTTPException(status_code=501, detail="模型下载功能待实现")


@app.get("/api/v1/training/model/versions")
async def list_model_versions():
    """获取可用模型版本列表"""
    # TODO: 实现版本历史管理
    return {"versions": [], "current": "yolov10n-v1.0.0"}


# ----- 难例收集接口 -----

@app.post("/api/v1/training/hard-example")
async def upload_hard_example(data: HardExampleUpload):
    """
    边缘上报难例到云端
    难例定义：YOLO置信度低(0.3-0.6) 或 SAEC复杂度高 的样本
    云端收集后用于增量学习和知识蒸馏
    """
    logger.info(f"📦 收到难例 | 边缘:{data.edge_id} 设备:{data.device_id} Sc={data.sc_score:.3f}")

    # TODO: 实现难例存储到数据库/对象存储
    # 1. 保存图像到存储 (如 MinIO/S3)
    # 2. 保存元数据到数据库
    # 3. 触发 nightly 训练流水线

    return {
        "status": "received",
        "example_id": f"HE-{int(time.time())}",
        "message": "难例已接收，将用于夜间增量训练"
    }


@app.get("/api/v1/training/hard-examples/stats")
async def get_hard_example_stats():
    """获取难例收集统计"""
    # TODO: 实现统计查询
    return {
        "total_collected": 0,
        "last_24h": 0,
        "by_edge": {},
        "pending_labeling": 0
    }


# ----- 知识蒸馏接口 -----

@app.post("/api/v1/training/distill", response_model=DistillationResponse)
async def get_distillation_labels(request: DistillationRequest):
    """
    获取大模型(Qwen3-VL)的软标签用于知识蒸馏
    边缘YOLO可以用这些软标签进行在线学习

    软标签 vs 硬标签：
    - 硬标签: [0, 0, 1, 0] (one-hot)
    - 软标签: [0.1, 0.2, 0.6, 0.1] (概率分布，更多信息)
    """
    logger.info(f"🎓 蒸馏请求 | 边缘:{request.edge_id} 温度T={request.temperature}")

    # TODO: 实现真实蒸馏逻辑
    # 1. 调用 Qwen3-VL 获取预测
    # 2. 应用温度参数生成软标签
    # 3. 返回软概率分布

    # 当前返回占位数据
    return DistillationResponse(
        soft_labels=[0.1, 0.1, 0.6, 0.2],  # 示例：4类分类的软标签
        boxes=[],
        feature_map=None,
        teacher_confidence=0.85
    )


@app.post("/api/v1/training/distill/batch")
async def batch_distillation(files: List[UploadFile] = File(...)):
    """
    批量获取蒸馏标签
    用于夜间批量训练
    """
    # TODO: 批量处理实现
    return {"status": "pending", "batch_id": f"BATCH-{int(time.time())}"}


# ----- 训练任务管理接口 -----

@app.post("/api/v1/training/task")
async def create_training_task(task: TrainingTask):
    """
    创建训练任务
    支持: distillation(知识蒸馏), incremental(增量学习), finetune(全量微调)
    """
    logger.info(f"🚀 创建训练任务 | 类型:{task.task_type} 版本:{task.model_version}")

    # TODO: 实现任务队列管理 (如使用 Celery + Redis)
    return {
        "task_id": task.task_id,
        "status": "pending",
        "message": "任务已加入队列"
    }


@app.get("/api/v1/training/task/{task_id}")
async def get_training_task_status(task_id: str):
    """查询训练任务状态"""
    # TODO: 实现任务状态查询
    return {
        "task_id": task_id,
        "status": "pending",
        "progress": 0.0,
        "logs": []
    }


@app.get("/api/v1/training/tasks")
async def list_training_tasks():
    """列出所有训练任务"""
    # TODO: 实现任务列表查询
    return {"tasks": [], "total": 0}


@app.delete("/api/v1/training/task/{task_id}")
async def cancel_training_task(task_id: str):
    """取消训练任务"""
    # TODO: 实现任务取消
    return {"task_id": task_id, "status": "cancelled"}


# ----- 增量学习反馈接口 -----

@app.post("/api/v1/training/feedback")
async def submit_learning_feedback(
    edge_id: str = Form(...),
    device_id: str = Form(...),
    model_version: str = Form(...),
    feedback_type: str = Form(...),  # "false_positive", "false_negative", "correct"
    image: UploadFile = File(...),
    annotation: str = Form("{}")  # JSON格式标注
):
    """
    边缘反馈模型预测结果
    用于持续改进模型质量
    """
    logger.info(f"💬 训练反馈 | 边缘:{edge_id} 类型:{feedback_type}")

    # TODO: 实现反馈收集
    # 1. 保存反馈图像和标注
    # 2. 更新模型评估指标
    # 3. 触发模型再训练（当反馈积累到一定量）

    return {
        "feedback_id": f"FB-{int(time.time())}",
        "status": "recorded"
    }


@app.get("/api/v1/training/metrics/{edge_id}")
async def get_edge_training_metrics(edge_id: str):
    """
    获取边缘设备的训练指标
    用于监控边缘模型性能退化
    """
    # TODO: 实现指标收集和查询
    return {
        "edge_id": edge_id,
        "precision": 0.0,
        "recall": 0.0,
        "mAP50": 0.0,
        "inference_time_ms": 0.0,
        "last_updated": time.time()
    }


# ----- 配置管理接口 -----

@app.get("/api/v1/training/config/{edge_id}")
async def get_edge_training_config(edge_id: str):
    """
    获取边缘训练配置
    云端统一下发训练超参数
    """
    # TODO: 实现配置管理
    return {
        "edge_id": edge_id,
        "learning_rate": 0.001,
        "batch_size": 16,
        "epochs": 10,
        "distillation_alpha": 0.7,  # 蒸馏损失权重
        "temperature": 2.0,
        "update_interval_hours": 24
    }


@app.post("/api/v1/training/config/{edge_id}")
async def update_edge_training_config(edge_id: str, config: Dict):
    """更新边缘训练配置"""
    # TODO: 实现配置更新
    return {"status": "updated", "edge_id": edge_id}


# ================= 前端 Dashboard =================

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>边云协同智能巡检系统</title>
    <link rel="icon" type="image/png" href="https://img.icons8.com/fluency/48/cloud.png">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        body {
            background-color: #f5f5f5;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        /* 侧边栏样式 - 参考sample.html */
        .sidebar {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
            height: 90vh;
            overflow-y: auto;
        }

        .sidebar h5 {
            margin-bottom: 15px;
            border-bottom: 1px solid #eaeaea;
            padding-bottom: 10px;
            color: #0d6efd;
        }

        /* 边缘服务器卡片 */
        .edge-server-card {
            background: white;
            border-radius: 10px;
            border: 1px solid #e0e0e0;
            margin-bottom: 15px;
            overflow: hidden;
            transition: all 0.3s;
        }

        .edge-server-card:hover {
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }

        .edge-server-header {
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .edge-server-header.has-alert {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }

        .edge-server-header.offline {
            background: #6c757d;
        }

        .edge-status-badge {
            padding: 4px 10px;
            border-radius: 15px;
            font-size: 12px;
            font-weight: 600;
            background: rgba(255,255,255,0.2);
        }

        /* 端侧设备列表 */
        .device-list-container {
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease-out;
        }

        .device-list-container.expanded {
            max-height: 2000px;
            transition: max-height 0.5s ease-in;
        }

        .device-item {
            border-bottom: 1px solid #f0f0f0;
        }

        .device-item:last-child {
            border-bottom: none;
        }

        .device-header {
            padding: 12px 15px;
            background: #fafafa;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: background 0.2s;
        }

        .device-header:hover {
            background: #f0f0f0;
        }

        .device-header.has-alert {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
        }

        .device-name {
            font-weight: 600;
            color: #333;
        }

        .device-status {
            font-size: 12px;
            padding: 3px 10px;
            border-radius: 12px;
        }

        .device-status.online {
            background: #d4edda;
            color: #155724;
        }

        .device-status.offline {
            background: #f8d7da;
            color: #721c24;
        }

        .device-status.warning {
            background: #fff3cd;
            color: #856404;
        }

        /* 报警记录列表 */
        .alert-list-container {
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease-out;
            background: white;
        }

        .alert-list-container.expanded {
            max-height: 1000px;
            transition: max-height 0.4s ease-in;
        }

        .alert-item {
            padding: 12px 15px 12px 40px;
            border-bottom: 1px solid #f5f5f5;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 12px;
            transition: background 0.2s;
        }

        .alert-item:hover {
            background: #f8f9fa;
        }

        .alert-item:last-child {
            border-bottom: none;
        }

        .alert-thumb {
            width: 60px;
            height: 45px;
            object-fit: cover;
            border-radius: 5px;
        }

        .alert-info {
            flex: 1;
        }

        .alert-title {
            font-size: 13px;
            font-weight: 600;
            margin-bottom: 3px;
        }

        .alert-time {
            font-size: 11px;
            color: #6c757d;
        }

        .alert-badge {
            padding: 3px 8px;
            border-radius: 10px;
            font-size: 11px;
            font-weight: 600;
        }

        .alert-badge.danger {
            background: #f8d7da;
            color: #721c24;
        }

        .alert-badge.warning {
            background: #fff3cd;
            color: #856404;
        }

        .alert-badge.normal {
            background: #d4edda;
            color: #155724;
        }

        /* 展开/收起图标 */
        .expand-icon {
            transition: transform 0.3s;
        }

        .expand-icon.rotated {
            transform: rotate(180deg);
        }

        /* 主内容区 */
        .main-container {
            background-color: #fff;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
            height: 90vh;
            display: flex;
            flex-direction: column;
        }

        .main-header {
            padding: 20px;
            border-bottom: 1px solid #eaeaea;
            background-color: #f8f9fa;
            border-top-left-radius: 15px;
            border-top-right-radius: 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .main-header h4 {
            margin: 0;
            color: #333;
        }

        .ai-status-badge {
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
        }

        .ai-online {
            background: rgba(40, 167, 69, 0.1);
            color: #28a745;
            border: 1px solid #28a745;
        }

        .ai-offline {
            background: rgba(220, 53, 69, 0.1);
            color: #dc3545;
            border: 1px solid #dc3545;
        }

        /* 统计卡片 */
        .stats-row {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            padding: 20px;
            border-bottom: 1px solid #eaeaea;
        }

        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #e0e0e0;
            text-align: center;
        }

        .stat-label {
            font-size: 13px;
            color: #6c757d;
            margin-bottom: 8px;
        }

        .stat-value {
            font-size: 28px;
            font-weight: bold;
            color: #333;
        }

        .stat-value.danger {
            color: #dc3545;
        }

        .stat-value.primary {
            color: #0d6efd;
        }

        /* 最近报警网格 */
        .alerts-grid {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 15px;
        }

        .alert-card {
            background: white;
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid #e0e0e0;
            transition: all 0.2s;
            cursor: pointer;
        }

        .alert-card:hover {
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            transform: translateY(-2px);
        }

        .alert-card-thumb {
            width: 100%;
            height: 160px;
            object-fit: cover;
            display: block;
        }

        .alert-card-body {
            padding: 15px;
        }

        .alert-card-badge {
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            display: inline-block;
        }

        .alert-card-badge.danger {
            background: #f8d7da;
            color: #721c24;
        }

        .alert-card-badge.warning {
            background: #fff3cd;
            color: #856404;
        }

        .alert-card-badge.normal {
            background: #d4edda;
            color: #155724;
        }

        .box-indicator {
            display: inline-block;
            margin-left: 8px;
            color: #0d6efd;
            font-size: 12px;
        }

        .alert-card-title {
            margin-top: 10px;
            font-weight: 600;
            font-size: 14px;
            color: #333;
        }

        .alert-card-time {
            margin-top: 6px;
            font-size: 12px;
            color: #6c757d;
        }

        .alert-card-source {
            margin-top: 8px;
            font-size: 11px;
            color: #0d6efd;
        }

        /* 空状态 */
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: #6c757d;
        }

        .empty-state i {
            font-size: 48px;
            margin-bottom: 15px;
            color: #dee2e6;
        }

        /* Modal & Canvas - 对比视图布局 */
        #modal {
            display: none;
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.95);
            z-index: 1000;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }

        /* 对比视图容器 */
        #canvas-compare-wrap {
            display: flex;
            gap: 20px;
            max-width: 95vw;
            max-height: 65vh;
            align-items: flex-start;
        }

        .canvas-panel {
            display: flex;
            flex-direction: column;
            align-items: center;
            background: #000;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 0 30px rgba(0,0,0,0.8);
        }

        .canvas-label {
            width: 100%;
            padding: 10px 15px;
            background: linear-gradient(135deg, #1a1a2e 0%, #2d2d44 100%);
            color: #fff;
            font-weight: 600;
            font-size: 14px;
            text-align: center;
            border-bottom: 1px solid #333;
        }

        .canvas-container {
            position: relative;
            max-width: 45vw;
            max-height: 55vh;
        }

        #yoloCanvas, #llmCanvas {
            display: block;
            max-width: 100%;
            max-height: 55vh;
        }

        .info-box {
            width: 100%;
            max-width: 900px;
            background: #1a1a2e;
            margin-top: 20px;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #333;
            max-height: 25vh;
            overflow-y: auto;
        }

        .close-btn {
            position: absolute;
            top: 20px;
            right: 30px;
            color: white;
            font-size: 40px;
            cursor: pointer;
            z-index: 1001;
            text-shadow: 0 2px 4px rgba(0,0,0,0.5);
        }

        @media (max-width: 768px) {
            .sidebar {
                height: auto;
                margin-bottom: 15px;
                max-height: 50vh;
            }
            .main-container {
                height: auto;
                min-height: 60vh;
            }
            .stats-row {
                grid-template-columns: repeat(2, 1fr);
            }
        }
    </style>
</head>
<body>
    <div class="container-fluid py-4">
        <div class="row">
            <!-- 侧边栏 - 边缘服务器层级结构 -->
            <div class="col-md-4 col-lg-3 mb-3">
                <div class="sidebar">
                    <h5>
                        <i class="fas fa-network-wired me-2"></i>
                        边缘-端侧设备拓扑
                    </h5>
                    <div id="edge-servers-container">
                        <div class="text-center text-muted py-4">
                            <i class="fas fa-spinner fa-spin fa-2x mb-2"></i>
                            <p>加载中...</p>
                        </div>
                    </div>
                </div>
            </div>

            <!-- 主内容区 - 报警详情 -->
            <div class="col-md-8 col-lg-9">
                <div class="main-container">
                    <div class="main-header">
                        <div>
                            <h4><i class="fas fa-cloud me-2 text-primary"></i>云端智能巡检中心</h4>
                            <small class="text-muted">基于 Qwen3-VL 的多模态分析</small>
                        </div>
                        <div id="ai-status" class="ai-status-badge ai-offline">
                            <i class="fas fa-circle me-1"></i>AI平台检测中...
                        </div>
                    </div>

                    <!-- 统计面板 -->
                    <div class="stats-row">
                        <div class="stat-card">
                            <div class="stat-label"><i class="fas fa-server me-1"></i>边缘节点</div>
                            <div class="stat-value" id="stat-edges">-</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-label"><i class="fas fa-mobile-alt me-1"></i>在线设备</div>
                            <div class="stat-value" id="stat-devices">-</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-label"><i class="fas fa-exclamation-triangle me-1"></i>异常告警</div>
                            <div class="stat-value danger" id="stat-alerts">-</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-label"><i class="fas fa-crosshairs me-1"></i>今日定位</div>
                            <div class="stat-value primary" id="stat-boxes">-</div>
                        </div>
                    </div>

                    <!-- 报警网格 -->
                    <div class="alerts-grid" id="alerts-grid">
                        <div class="empty-state" style="grid-column: 1 / -1;">
                            <i class="fas fa-spinner fa-spin"></i>
                            <p>正在加载数据...</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Modal - 对比视图：左侧YOLO，右侧Qwen3-VL -->
    <div id="modal" onclick="if(event.target===this)closeModal()">
        <span class="close-btn" onclick="closeModal()">&times;</span>

        <!-- 对比画布区域 -->
        <div id="canvas-compare-wrap">
            <div class="canvas-panel">
                <div class="canvas-label" id="yolo-info">
                    <span style="color:#ff4444"><i class="fas fa-robot"></i> 边缘YOLO</span>
                </div>
                <div class="canvas-container">
                    <canvas id="yoloCanvas"></canvas>
                </div>
            </div>
            <div class="canvas-panel">
                <div class="canvas-label" id="llm-info">
                    <span style="color:#00ff00"><i class="fas fa-brain"></i> 云端Qwen3-VL</span>
                </div>
                <div class="canvas-container">
                    <canvas id="llmCanvas"></canvas>
                </div>
            </div>
        </div>

        <!-- 信息区域 -->
        <div class="info-box">
            <h4 id="m-title" style="margin-top:0; color:#58a6ff;"></h4>
            <p id="m-text" style="line-height:1.6; color:#c9d1d9; white-space:pre-wrap;"></p>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // 全局数据存储
        let allRecords = [];
        let allDevices = [];
        let totalBoxes = 0;
        let isFirstLoad = true;
        
        // 保存展开状态
        const expandedEdges = new Set();
        const expandedDevices = new Set();

        // 组织数据结构 - 包含边缘服务器状态
        function organizeData(devices, records, edgesData) {
            const edgeMap = {};
            
            // 先初始化边缘服务器数据
            edgesData.forEach(edge => {
                edgeMap[edge.edge_id] = {
                    edge_id: edge.edge_id,
                    status: edge.status || 'offline',
                    last_heartbeat: edge.last_heartbeat,
                    devices: [],
                    hasAlert: false,
                    offlineCount: 0,
                    device_count: edge.device_count || 0
                };
            });
            
            // 将设备分配到对应边缘
            devices.forEach(device => {
                const edgeId = device.edge_id || 'UNKNOWN';
                if (!edgeMap[edgeId]) {
                    edgeMap[edgeId] = {
                        edge_id: edgeId,
                        status: 'unknown',
                        devices: [],
                        hasAlert: false,
                        offlineCount: 0
                    };
                }
                edgeMap[edgeId].devices.push(device);
                if (device.status === 'offline') {
                    edgeMap[edgeId].offlineCount++;
                }
            });

            // 为每个设备关联报警记录
            Object.values(edgeMap).forEach(edge => {
                edge.devices.forEach(device => {
                    device.records = records.filter(r => r.device_id === device.device_id);
                    device.hasAlert = device.records.some(r => r.alert_level !== 'normal');
                    device.alertCount = device.records.filter(r => r.alert_level !== 'normal').length;
                });
                edge.hasAlert = edge.devices.some(d => d.hasAlert);
            });

            return Object.values(edgeMap);
        }

        // 保存当前展开状态
        function saveExpandedState() {
            document.querySelectorAll('.device-list-container.expanded').forEach(el => {
                const edgeId = el.id.replace('edge-devices-', '');
                expandedEdges.add(edgeId);
            });
            document.querySelectorAll('.alert-list-container.expanded').forEach(el => {
                const id = el.id.replace('alerts-', '');
                expandedDevices.add(id);
            });
        }

        // 恢复展开状态
        function restoreExpandedState() {
            expandedEdges.forEach(edgeId => {
                const container = document.getElementById(`edge-devices-${edgeId}`);
                const icon = document.getElementById(`edge-icon-${edgeId}`);
                if (container && icon) {
                    container.classList.add('expanded');
                    icon.classList.add('rotated');
                }
            });
            expandedDevices.forEach(deviceKey => {
                const container = document.getElementById(`alerts-${deviceKey}`);
                const icon = document.getElementById(`device-icon-${deviceKey}`);
                if (container && icon) {
                    container.classList.add('expanded');
                    icon.classList.add('rotated');
                }
            });
        }

        // 渲染边缘服务器层级结构
        function renderEdgeServers(edges) {
            const container = document.getElementById('edge-servers-container');
            
            if (edges.length === 0) {
                container.innerHTML = `
                    <div class="text-center text-muted py-4">
                        <i class="fas fa-inbox fa-2x mb-2"></i>
                        <p>暂无边缘节点</p>
                    </div>
                `;
                return;
            }

            container.innerHTML = edges.map((edge) => {
                const deviceCount = edge.devices.length;
                const reportedCount = edge.device_count || deviceCount;
                const alertCount = edge.devices.reduce((sum, d) => sum + d.alertCount, 0);
                const hasAlert = edge.hasAlert;
                const isEdgeOffline = edge.status === 'offline';
                const allDevicesOffline = edge.offlineCount === deviceCount && deviceCount > 0;
                const showAsOffline = isEdgeOffline || allDevicesOffline;

                return `
                    <div class="edge-server-card">
                        <div class="edge-server-header ${hasAlert ? 'has-alert' : ''} ${showAsOffline ? 'offline' : ''}" 
                             onclick="toggleEdge('${edge.edge_id}')">
                            <div>
                                <i class="fas fa-server me-2"></i>
                                <strong>${edge.edge_id}</strong>
                                <span class="ms-2" style="font-size: 12px; opacity: 0.9;">
                                    (${reportedCount}个端侧)
                                </span>
                                ${isEdgeOffline ? '<span class="badge bg-secondary ms-1">边缘离线</span>' : ''}
                            </div>
                            <div class="d-flex align-items-center gap-2">
                                ${alertCount > 0 ? `<span class="badge bg-warning text-dark"><i class="fas fa-bell me-1"></i>${alertCount}</span>` : ''}
                                ${showAsOffline ? '<span class="edge-status-badge">离线</span>' : '<span class="edge-status-badge">在线</span>'}
                                <i class="fas fa-chevron-down expand-icon" id="edge-icon-${edge.edge_id}"></i>
                            </div>
                        </div>
                        <div class="device-list-container" id="edge-devices-${edge.edge_id}">
                            ${edge.devices.map((device) => `
                                <div class="device-item">
                                    <div class="device-header ${device.hasAlert ? 'has-alert' : ''}" 
                                         onclick="toggleDevice('${edge.edge_id}', '${device.device_id}', event)">
                                        <div class="d-flex align-items-center">
                                            <i class="fas fa-mobile-alt me-2 ${device.status === 'online' ? 'text-success' : 'text-danger'}"></i>
                                            <span class="device-name">${device.device_id}</span>
                                        </div>
                                        <div class="d-flex align-items-center gap-2">
                                            ${device.alertCount > 0 ? `<span class="badge bg-danger">${device.alertCount} 告警</span>` : ''}
                                            <span class="device-status ${device.status}">${device.status === 'online' ? '在线' : '离线'}</span>
                                            <i class="fas fa-chevron-down expand-icon text-muted" 
                                               id="device-icon-${edge.edge_id}-${device.device_id}"></i>
                                        </div>
                                    </div>
                                    <div class="alert-list-container" id="alerts-${edge.edge_id}-${device.device_id}">
                                        ${device.records.length === 0 ? `
                                            <div class="alert-item text-muted">
                                                <i class="fas fa-check-circle me-2 text-success"></i>
                                                暂无报警记录
                                            </div>
                                        ` : device.records.map(record => `
                                            <div class="alert-item" onclick="showDetailFromElement(this, event)" 
                                                 data-record='${JSON.stringify(record).replace(/'/g, "&#39;")}'>
                                                <img src="${record.image_url}" class="alert-thumb" alt="">
                                                <div class="alert-info">
                                                    <div class="alert-title">${record.trigger_reason}</div>
                                                    <div class="alert-time">${record.timestamp}</div>
                                                </div>
                                                <span class="alert-badge ${record.alert_level}">
                                                    ${record.alert_level === 'danger' ? '🔥 危险' : 
                                                      record.alert_level === 'warning' ? '⚠️ 异常' : '✓ 正常'}
                                                </span>
                                            </div>
                                        `).join('')}
                                    </div>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                `;
            }).join('');
        }

        // 切换边缘服务器展开/收起
        function toggleEdge(edgeId) {
            const container = document.getElementById(`edge-devices-${edgeId}`);
            const icon = document.getElementById(`edge-icon-${edgeId}`);
            const isExpanded = container.classList.toggle('expanded');
            icon.classList.toggle('rotated');
            
            // 更新状态集合
            if (isExpanded) {
                expandedEdges.add(edgeId);
            } else {
                expandedEdges.delete(edgeId);
            }
        }

        // 切换设备展开/收起
        function toggleDevice(edgeId, deviceId, event) {
            event.stopPropagation();
            const container = document.getElementById(`alerts-${edgeId}-${deviceId}`);
            const icon = document.getElementById(`device-icon-${edgeId}-${deviceId}`);
            const isExpanded = container.classList.toggle('expanded');
            icon.classList.toggle('rotated');
            
            // 更新状态集合
            const deviceKey = `${edgeId}-${deviceId}`;
            if (isExpanded) {
                expandedDevices.add(deviceKey);
            } else {
                expandedDevices.delete(deviceKey);
            }
        }

        // 从元素显示详情
        function showDetailFromElement(element, event) {
            event.stopPropagation();
            const recordData = element.getAttribute('data-record');
            const record = JSON.parse(recordData);
            showDetail(record);
        }

        // 渲染主区域报警卡片 - 使用增量更新
        function renderAlertCards(records) {
            const grid = document.getElementById('alerts-grid');
            
            if (records.length === 0) {
                grid.innerHTML = `
                    <div class="empty-state" style="grid-column: 1 / -1;">
                        <i class="fas fa-check-circle"></i>
                        <h5>暂无检测记录</h5>
                        <p>系统运行正常，未发现异常</p>
                    </div>
                `;
                return;
            }

            // 第一次加载或记录数量变化时全量渲染，否则增量更新
            if (isFirstLoad || grid.children.length !== records.length) {
                grid.innerHTML = records.map(r => createAlertCard(r)).join('');
            } else {
                // 只更新状态可能变化的卡片
                records.forEach((r, index) => {
                    const card = grid.children[index];
                    if (card) {
                        updateAlertCard(card, r);
                    }
                });
            }
        }

        // 创建报警卡片HTML
        function createAlertCard(r) {
            const levelClass = r.alert_level === 'danger' ? 'danger' : 
                              r.alert_level === 'warning' ? 'warning' : 'normal';
            const levelText = r.alert_level === 'danger' ? '🔥 危险' : 
                             r.alert_level === 'warning' ? '⚠️ 异常' : '✓ 正常';
            const boxText = r.boxes.length > 0 ? `<span class="box-indicator"><i class="fas fa-crosshairs me-1"></i>${r.boxes.length}个定位</span>` : '';
            
            return `
                <div class="alert-card" onclick='showDetail(${JSON.stringify(r).replace(/"/g, '&quot;')})'>
                    <img class="alert-card-thumb" src="${r.image_url}" alt="检测图像" loading="lazy">
                    <div class="alert-card-body">
                        <div class="d-flex justify-content-between align-items-start">
                            <span class="alert-card-badge ${levelClass}">${levelText}</span>
                            ${boxText}
                        </div>
                        <div class="alert-card-title">${r.trigger_reason}</div>
                        <div class="alert-card-time"><i class="far fa-clock me-1"></i>${r.timestamp}</div>
                        <div class="alert-card-source">
                            <i class="fas fa-server me-1"></i>${r.edge_id} / ${r.device_id}
                        </div>
                    </div>
                </div>
            `;
        }

        // 更新报警卡片（不重新创建DOM）
        function updateAlertCard(card, r) {
            const badge = card.querySelector('.alert-card-badge');
            const newLevelClass = r.alert_level === 'danger' ? 'danger' : 
                                  r.alert_level === 'warning' ? 'warning' : 'normal';
            const newLevelText = r.alert_level === 'danger' ? '🔥 危险' : 
                                 r.alert_level === 'warning' ? '⚠️ 异常' : '✓ 正常';
            
            // 只在变化时更新
            if (!badge.classList.contains(newLevelClass)) {
                badge.className = `alert-card-badge ${newLevelClass}`;
                badge.textContent = newLevelText;
            }
        }

        // 显示详情模态框 - 对比视图：左侧YOLO，右侧Qwen3-VL
        function showDetail(r) {
            const modal = document.getElementById('modal');
            const leftCanvas = document.getElementById('yoloCanvas');
            const rightCanvas = document.getElementById('llmCanvas');
            const leftCtx = leftCanvas.getContext('2d');
            const rightCtx = rightCanvas.getContext('2d');
            const img = new Image();

            img.onload = () => {
                // 设置画布大小
                leftCanvas.width = img.width;
                leftCanvas.height = img.height;
                rightCanvas.width = img.width;
                rightCanvas.height = img.height;

                // 绘制原始图像
                leftCtx.drawImage(img, 0, 0);
                rightCtx.drawImage(img, 0, 0);

                // ===== 左侧：YOLO检测框（红色） =====
                leftCtx.strokeStyle = '#ff4444';
                leftCtx.lineWidth = Math.max(img.width / 200, 3);
                leftCtx.shadowBlur = 10;
                leftCtx.shadowColor = '#ff4444';

                if (r.yolo_boxes && r.yolo_boxes.length > 0) {
                    r.yolo_boxes.forEach((box, idx) => {
                        const [y1, x1, y2, x2] = box.box_norm;
                        const rx = (x1 / 1000) * img.width;
                        const ry = (y1 / 1000) * img.height;
                        const rw = ((x2 - x1) / 1000) * img.width;
                        const rh = ((y2 - y1) / 1000) * img.height;

                        leftCtx.strokeRect(rx, ry, rw, rh);

                        // 标签
                        leftCtx.fillStyle = '#ff4444';
                        leftCtx.font = `bold ${Math.max(14, img.width/40)}px Arial`;
                        const label = `YOLO-${idx+1} ${(box.conf * 100).toFixed(0)}%`;
                        leftCtx.fillText(label, rx + 4, ry - 6);
                    });
                } else {
                    // 无YOLO检测框提示
                    leftCtx.fillStyle = 'rgba(255, 255, 255, 0.8)';
                    leftCtx.fillRect(10, 10, 200, 30);
                    leftCtx.fillStyle = '#ff4444';
                    leftCtx.font = `bold 16px Arial`;
                    leftCtx.fillText('YOLO: 未检测到目标', 20, 32);
                }

                // ===== 右侧：Qwen3-VL检测框（绿色） =====
                rightCtx.strokeStyle = '#00ff00';
                rightCtx.lineWidth = Math.max(img.width / 200, 3);
                rightCtx.shadowBlur = 10;
                rightCtx.shadowColor = '#00ff00';

                if (r.boxes && r.boxes.length > 0) {
                    r.boxes.forEach((box, idx) => {
                        const [x1, y1, x2, y2] = box; // 关键：顺序为 [x1, y1, x2, y2]
                        const rx = (x1 / 1000) * img.width;
                        const ry = (y1 / 1000) * img.height;
                        const rw = ((x2 - x1) / 1000) * img.width;
                        const rh = ((y2 - y1) / 1000) * img.height;

                        rightCtx.strokeRect(rx, ry, rw, rh);

                        rightCtx.fillStyle = '#00ff00';
                        rightCtx.font = `bold ${Math.max(14, img.width/40)}px Arial`;
                        rightCtx.fillText(`LLM-${idx+1}`, rx + 4, ry - 6);
                    });
                } else {
                    // 无LLM检测框提示
                    rightCtx.fillStyle = 'rgba(255, 255, 255, 0.8)';
                    rightCtx.fillRect(10, 10, 200, 30);
                    rightCtx.fillStyle = '#00ff00';
                    rightCtx.font = `bold 16px Arial`;
                    rightCtx.fillText('Qwen3-VL: 未标注区域', 20, 32);
                }

                leftCtx.shadowBlur = 0;
                rightCtx.shadowBlur = 0;

                // 更新信息
                document.getElementById('m-title').innerText = `${r.trigger_reason} [${r.alert_level.toUpperCase()}]`;
                document.getElementById('m-text').innerText = r.llm_result || '无分析结果';

                // 更新对比信息
                const yoloCount = r.yolo_boxes ? r.yolo_boxes.length : 0;
                const llmCount = r.boxes ? r.boxes.length : 0;
                document.getElementById('yolo-info').innerHTML = `
                    <span style="color:#ff4444"><i class="fas fa-robot"></i> 边缘YOLO</span>
                    <br>检测到 ${yoloCount} 个目标
                `;
                document.getElementById('llm-info').innerHTML = `
                    <span style="color:#00ff00"><i class="fas fa-brain"></i> 云端Qwen3-VL</span>
                    <br>标注了 ${llmCount} 个区域
                `;

                modal.style.display = 'flex';
            };

            img.onerror = () => {
                alert('图像加载失败');
            };

            img.src = r.image_url;
        }

        function closeModal() {
            document.getElementById('modal').style.display = 'none';
        }

        // ESC关闭
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') closeModal();
        });

        // 更新数据
        async function update() {
            try {
                const [rRes, sRes, dRes, hRes] = await Promise.all([
                    fetch('/api/v1/records'),
                    fetch('/api/v1/statistics'),
                    fetch('/api/v1/devices'),
                    fetch('/api/v1/health')
                ]);
                
                const rData = await rRes.json();
                const sData = await sRes.json();
                const dData = await dRes.json();
                const hData = await hRes.json();

                allRecords = rData.records || [];
                allDevices = dData.devices || [];

                // 更新AI状态 - 修复显示
                const aiStatus = document.getElementById('ai-status');
                if (hData.swift_online) {
                    aiStatus.className = 'ai-status-badge ai-online';
                    aiStatus.innerHTML = '<i class="fas fa-circle me-1"></i>AI平台在线';
                } else {
                    aiStatus.className = 'ai-status-badge ai-offline';
                    aiStatus.innerHTML = '<i class="fas fa-circle me-1"></i>AI平台离线';
                }

                // 更新统计 - 使用API返回的边缘数据
                const onlineEdges = dData.summary ? dData.summary.online_edges : new Set(allDevices.map(d => d.edge_id)).size;
                const onlineDevices = dData.summary ? dData.summary.online_devices : sData.data.online_devices;
                document.getElementById('stat-edges').innerText = `${onlineEdges}/${sData.data.total_edges || onlineEdges}`;
                document.getElementById('stat-devices').innerText = `${onlineDevices}/${sData.data.total_devices || onlineDevices}`;
                document.getElementById('stat-alerts').innerText = sData.data.danger_alerts;
                
                totalBoxes = allRecords.reduce((sum, r) => sum + r.boxes.length, 0);
                document.getElementById('stat-boxes').innerText = totalBoxes;

                // 保存当前展开状态
                saveExpandedState();

                // 组织并渲染层级结构 - 传入边缘服务器数据
                const edges = organizeData(allDevices, allRecords, dData.edges || []);
                renderEdgeServers(edges);

                // 恢复展开状态
                restoreExpandedState();

                // 渲染主区域报警卡片
                renderAlertCards(allRecords);
                
                isFirstLoad = false;

            } catch(e) { 
                console.error('Update error:', e); 
            }
        }

        // 定时更新 - 改为5秒，减少刷新频率
        setInterval(update, 5000);
        update();
    </script>
</body>
</html>
    """

# ================= 守护任务 =================

async def check_swift_health():
    global swift_online
    while True:
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(SWIFT_HEALTH_URL, timeout=5) as r:
                    swift_online = (r.status == 200)
        except:
            swift_online = False
        await asyncio.sleep(10)


async def check_device_timeout():
    """定时检查设备报警超时，5分钟无报警置为normal"""
    while True:
        await asyncio.sleep(30)  # 每30秒检查一次
        state.check_device_status_timeout(timeout_seconds=300)  # 5分钟 = 300秒


async def check_edge_and_device_connection():
    """定时检查边缘服务器和设备连接状态"""
    while True:
        await asyncio.sleep(15)  # 每15秒检查一次
        state.check_edge_and_device_timeout()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=CLOUD_PORT)
