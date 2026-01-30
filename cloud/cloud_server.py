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
    status: str = "online"
    total_inspections: int = 0
    last_alert: Optional[Dict] = None

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
    boxes: List[List[int]] = field(default_factory=list)  # [y1, x1, y2, x2] 归一化坐标

# ================= 状态管理 =================

class StateManager:
    def __init__(self):
        self.devices: Dict[str, Device] = {}
        self.records: List[InspectionRecord] = []
        self.max_records = 100
        self._lock = Lock()

    def update_device(self, device_id: str, edge_id: str, status: str = "online"):
        with self._lock:
            key = f"{edge_id}:{device_id}"
            now = time.time()
            if key not in self.devices:
                self.devices[key] = Device(device_id=device_id, edge_id=edge_id, first_seen=now, last_seen=now)
            self.devices[key].last_seen = now
            self.devices[key].status = status

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
                "boxes": r.boxes
            } for r in self.records]

    def get_statistics(self):
        with self._lock:
            return {
                "total_devices": len(self.devices),
                "online_devices": sum(1 for d in self.devices.values() if d.status == "online"),
                "total_inspections": len(self.records),
                "danger_alerts": sum(1 for r in self.records if r.alert_level == "danger")
            }
    
    def get_devices(self):
        with self._lock:
            return list(self.devices.values())

state = StateManager()

# ================= 核心工具：Qwen3-VL 坐标解析 =================

def extract_qwen_boxes(text: str) -> List[List[int]]:
    """
    解析 Qwen3-VL 视觉定位坐标
    格式: <|box_start|>(y1,x1,y2,x2)<|box_end|>
    坐标范围: 0-1000 (归一化)
    顺序: [y1, x1, y2, x2] (注意 y 在前!)
    
    返回: List[[y1, x1, y2, x2], ...]
    """
    if not text:
        return []
    
    # 匹配 <|box_start|>(y1,x1,y2,x2)<|box_end|>
    pattern = r"<\|box_start\|>\((\d+),(\d+),(\d+),(\d+)\)<\|box_end\|>"
    matches = re.findall(pattern, text)
    
    # 转换为整数列表 [[y1, x1, y2, x2], ...]
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
    """构建带视觉定位指令的 Prompt"""
    
    base_instruction = """请作为电力设备巡检专家，仔细分析这张图像。

任务要求：
1. 检查是否存在异常、火情、设备损坏或安全隐患
2. 如果发现问题，请详细描述异常位置和特征
3. **重要：用坐标框出异常区域** 

坐标格式说明：
- 使用 <|box_start|>(y1,x1,y2,x2)<|box_end|> 标记位置
- 坐标范围 0-1000，y 在前 x 在后
- 例如：变压器漏油点 <|box_start|>(300,150,450,400)<|box_end|>

请用中文回复，包含：
1. 整体状态评估
2. 具体问题描述（如有）
3. 异常位置的坐标标记"""

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

# ================= API 接口 =================

app = FastAPI(title="Cloud Vision Inspection Center")
app.mount("/static", StaticFiles(directory=TEMP_DIR), name="static")

@app.post("/api/v1/inspect")
async def inspect(
    image: UploadFile = File(...),
    device_id: str = Form(...),
    edge_id: str = Form(...),
    trigger_reason: str = Form("unknown")
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
    
    # AI 分析
    if swift_online:
        prompt = build_grounding_prompt(trigger_reason)
        llm_raw = await call_swift_deploy(save_path, prompt)
    else:
        llm_raw = "AI算力平台离线，请人工复核"
    
    # 解析坐标
    boxes = extract_qwen_boxes(llm_raw)
    logger.info(f"🔍 检测到 {len(boxes)} 个定位框")
    
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
            boxes=boxes
        )
        state.add_record(record)
    
    return {
        "status": "success",
        "alert_level": alert_level,
        "boxes_count": len(boxes),
        "llm_result": llm_raw[:200] + "..." if len(llm_raw) > 200 else llm_raw
    }


@app.post("/api/v1/edge_heartbeat")
async def heartbeat(edge_id: str = Form(...), device_id: str = Form(...)):
    state.update_device(device_id, edge_id)
    return {"swift_online": swift_online, "status": "ok"}


@app.get("/api/v1/records")
async def get_records():
    return {"records": state.get_records_json()}


@app.get("/api/v1/statistics")
async def get_stats():
    return {"data": state.get_statistics()}


@app.get("/api/v1/devices")
async def get_devices():
    return {"devices": [
        {**d.__dict__, "first_seen": datetime.fromtimestamp(d.first_seen).isoformat(),
         "last_seen": datetime.fromtimestamp(d.last_seen).isoformat()}
        for d in state.get_devices()
    ]}


@app.get("/api/v1/health")
async def health():
    return {"swift_online": swift_online, "status": "healthy"}


# ================= 前端 Dashboard =================

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>云端智能巡检指挥中心</title>
    <style>
        :root { 
            --bg: #0b0e14; 
            --card: #161b22; 
            --primary: #58a6ff; 
            --danger: #f85149; 
            --warning: #d29922;
            --success: #3fb950;
            --border: #30363d;
        }
        body { 
            background: var(--bg); 
            color: #c9d1d9; 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; 
            margin: 0; 
            padding: 20px;
        }
        .header { 
            display: flex; 
            justify-content: space-between; 
            align-items: center; 
            border-bottom: 1px solid var(--border); 
            padding-bottom: 15px; 
            margin-bottom: 25px;
        }
        .header h1 { margin: 0; font-size: 24px; }
        .swift-tag { 
            padding: 6px 14px; 
            border-radius: 20px; 
            font-size: 12px; 
            font-weight: 600;
        }
        .swift-online { background: rgba(63,185,80,0.2); color: var(--success); border: 1px solid var(--success); }
        .swift-offline { background: rgba(248,81,73,0.2); color: var(--danger); border: 1px solid var(--danger); }
        
        .stats-bar { 
            display: grid; 
            grid-template-columns: repeat(4, 1fr); 
            gap: 15px; 
            margin-bottom: 25px; 
        }
        .stat-card { 
            background: var(--card); 
            padding: 20px; 
            border-radius: 8px; 
            border: 1px solid var(--border);
        }
        .stat-label { font-size: 13px; color: #8b949e; margin-bottom: 8px; }
        .stat-val { font-size: 28px; font-weight: bold; color: white; }
        
        .main-grid { 
            display: grid; 
            grid-template-columns: 2fr 1fr; 
            gap: 20px; 
        }
        .record-list { 
            display: grid; 
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); 
            gap: 15px; 
        }
        .card { 
            background: var(--card); 
            border-radius: 10px; 
            overflow: hidden; 
            border: 1px solid var(--border); 
            transition: all 0.2s;
            cursor: pointer;
        }
        .card:hover { 
            border-color: var(--primary); 
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        .thumb { 
            width: 100%; 
            height: 160px; 
            object-fit: cover; 
            display: block;
        }
        .card-body { padding: 14px; }
        .badge { 
            padding: 3px 10px; 
            border-radius: 12px; 
            font-size: 11px; 
            font-weight: bold;
            display: inline-block;
        }
        .bg-danger { background: rgba(248,81,73,0.2); color: var(--danger); border: 1px solid var(--danger); }
        .bg-warning { background: rgba(210,153,34,0.2); color: var(--warning); border: 1px solid var(--warning); }
        .bg-success { background: rgba(63,185,80,0.2); color: var(--success); border: 1px solid var(--success); }
        .box-indicator { 
            display: inline-block; 
            margin-left: 8px; 
            color: var(--primary); 
            font-size: 12px;
        }

        /* Modal & Canvas */
        #modal { 
            display: none; 
            position: fixed; 
            inset: 0; 
            background: rgba(0,0,0,0.95); 
            z-index: 100; 
            flex-direction: column; 
            align-items: center; 
            justify-content: center;
            padding: 20px;
        }
        #canvas-wrap { 
            position: relative; 
            background: #000; 
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 0 30px rgba(0,0,0,0.8);
            max-width: 90vw;
            max-height: 70vh;
        }
        canvas { 
            display: block;
            max-width: 100%;
            max-height: 70vh;
        }
        .info-box { 
            width: 100%;
            max-width: 900px;
            background: var(--card); 
            margin-top: 20px; 
            padding: 20px; 
            border-radius: 10px;
            border: 1px solid var(--border);
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
            z-index: 101;
            text-shadow: 0 2px 4px rgba(0,0,0,0.5);
        }
        .device-list {
            background: var(--card);
            border-radius: 10px;
            border: 1px solid var(--border);
            padding: 15px;
        }
        .device-item {
            padding: 12px 0;
            border-bottom: 1px solid var(--border);
        }
        .device-item:last-child { border-bottom: none; }
    </style>
</head>
<body>
    <div class="header">
        <h1>☁️ Cloud Vision Center - Qwen3-VL 智能巡检</h1>
        <div id="swift-tag" class="swift-tag swift-offline">AI平台检测中...</div>
    </div>

    <div class="stats-bar">
        <div class="stat-card">
            <div class="stat-label">在线设备</div>
            <div class="stat-val" id="s-online">-</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">累计检测</div>
            <div class="stat-val" id="s-total">-</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">异常告警</div>
            <div class="stat-val" id="s-danger" style="color:var(--danger)">-</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">今日定位框</div>
            <div class="stat-val" id="s-boxes" style="color:var(--primary)">-</div>
        </div>
    </div>

    <div class="main-grid">
        <div class="record-list" id="record-grid"></div>
        <div class="device-list">
            <h3 style="margin-top:0; margin-bottom:15px; color:#8b949e; font-size:14px;">📱 在线设备</h3>
            <div id="device-list"></div>
        </div>
    </div>

    <!-- Modal -->
    <div id="modal" onclick="if(event.target===this)closeModal()">
        <span class="close-btn" onclick="closeModal()">&times;</span>
        <div id="canvas-wrap">
            <canvas id="mainCanvas"></canvas>
        </div>
        <div class="info-box">
            <h3 id="m-title" style="margin-top:0; color:var(--primary);"></h3>
            <p id="m-text" style="line-height:1.6; color:#c9d1d9; white-space:pre-wrap;"></p>
        </div>
    </div>

    <script>
        let totalBoxes = 0;

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

                // 更新AI状态
                const swiftTag = document.getElementById('swift-tag');
                if (hData.swift_online) {
                    swiftTag.className = 'swift-tag swift-online';
                    swiftTag.innerText = 'AI平台在线';
                } else {
                    swiftTag.className = 'swift-tag swift-offline';
                    swiftTag.innerText = 'AI平台离线';
                }

                // 更新统计
                document.getElementById('s-online').innerText = sData.data.online_devices;
                document.getElementById('s-total').innerText = sData.data.total_inspections;
                document.getElementById('s-danger').innerText = sData.data.danger_alerts;

                // 计算总框数
                totalBoxes = rData.records.reduce((sum, r) => sum + r.boxes.length, 0);
                document.getElementById('s-boxes').innerText = totalBoxes;

                // 更新记录卡片
                document.getElementById('record-grid').innerHTML = rData.records.map(r => {
                    const levelClass = r.alert_level === 'danger' ? 'bg-danger' : 
                                      r.alert_level === 'warning' ? 'bg-warning' : 'bg-success';
                    const levelText = r.alert_level === 'danger' ? '🔥 危险' : 
                                     r.alert_level === 'warning' ? '⚠️ 异常' : '✓ 正常';
                    const boxText = r.boxes.length > 0 ? `<span class="box-indicator">📍 ${r.boxes.length}个定位框</span>` : '';
                    
                    return `
                        <div class="card" onclick='showDetail(${JSON.stringify(r)})'>
                            <img class="thumb" src="${r.image_url}" alt="检测图像">
                            <div class="card-body">
                                <span class="badge ${levelClass}">${levelText}</span>
                                ${boxText}
                                <div style="margin-top:10px; font-weight:600; font-size:14px;">${r.trigger_reason}</div>
                                <div style="margin-top:6px; font-size:12px; color:#8b949e;">${r.timestamp}</div>
                            </div>
                        </div>
                    `;
                }).join('') || '<div style="color:#8b949e; padding:40px;">暂无检测记录</div>';

                // 更新设备列表
                document.getElementById('device-list').innerHTML = dData.devices.map(d => `
                    <div class="device-item">
                        <div style="display:flex; justify-content:space-between;">
                            <b>${d.device_id}</b>
                            <span style="font-size:12px; color:${d.status==='online'?'var(--success)':'var(--danger)'};">${d.status}</span>
                        </div>
                        <div style="font-size:12px; color:#8b949e; margin-top:4px;">
                            检测 ${d.total_inspections} 次 | ${d.edge_id}
                        </div>
                    </div>
                `).join('') || '<div style="color:#8b949e;">暂无设备</div>';

            } catch(e) { console.error(e); }
        }

        function showDetail(r) {
            const modal = document.getElementById('modal');
            const canvas = document.getElementById('mainCanvas');
            const ctx = canvas.getContext('2d');
            const img = new Image();
            
            img.onload = () => {
                // 设置 canvas 尺寸
                canvas.width = img.width;
                canvas.height = img.height;
                
                // 绘制原图
                ctx.drawImage(img, 0, 0);
                
                // 绘制检测框
                ctx.strokeStyle = '#00ff00';
                ctx.lineWidth = Math.max(img.width / 200, 3);
                ctx.shadowBlur = 10;
                ctx.shadowColor = '#00ff00';
                
                r.boxes.forEach((box, idx) => {
                    const [y1, x1, y2, x2] = box;
                    // Qwen3-VL 坐标是 0-1000 归一化，y在前x在后
                    const rx = (x1 / 1000) * img.width;
                    const ry = (y1 / 1000) * img.height;
                    const rw = ((x2 - x1) / 1000) * img.width;
                    const rh = ((y2 - y1) / 1000) * img.height;
                    
                    ctx.strokeRect(rx, ry, rw, rh);
                    
                    // 绘制编号
                    ctx.fillStyle = '#00ff00';
                    ctx.font = `bold ${Math.max(14, img.width/40)}px Arial`;
                    ctx.fillText(`#${idx+1}`, rx + 4, ry - 6);
                });
                
                // 恢复样式
                ctx.shadowBlur = 0;

                // 显示信息
                document.getElementById('m-title').innerText = `${r.trigger_reason} [${r.alert_level.toUpperCase()}]`;
                document.getElementById('m-text').innerText = r.llm_result || '无分析结果';
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
        
        // ESC 关闭
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') closeModal();
        });
        
        setInterval(update, 3000);
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

@app.on_event("startup")
async def startup():
    asyncio.create_task(check_swift_health())
    logger.info(f"☁️ 云端视觉巡检中心启动 | 端口:{CLOUD_PORT}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=CLOUD_PORT)
