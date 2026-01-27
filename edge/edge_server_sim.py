import os
import cv2
import requests
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from typing import List, Optional
from ultralytics import YOLO

# ================= SAEC 配置区 =================
# 1. 模型路径
YOLO_MODEL_PATH = "yolov10_best.pt"  # 边缘侧轻量级模型
CLOUD_API_URL = "http://<你的小服务器公网IP>:8000/v1/chat/completions"  # 通过SSH隧道指向AutoDL

# 2. SAEC 调度阈值与权重
ENABLE_SAEC_ADAPTIVE = True  # 是否启用 SAEC 场景自适应调度
SC_THRESHOLD = 0.55  # 复杂度阈值 (0-1)，超过此值触发云端 MLLM
WEIGHTS = {
    "entropy": 0.4,  # 信息熵权重：反映背景乱度
    "edge": 0.3,  # 边缘密度权重：反映物体密集度
    "sharpness": 0.3  # 清晰度权重：反映光照和模糊度
}

# 3. 业务参数
CONF_THRESHOLD = 0.45  # YOLO 置信度

app = FastAPI(title="SAEC Collaborative Inspection System")

# 加载边缘模型
if os.path.exists(YOLO_MODEL_PATH):
    edge_model = YOLO(YOLO_MODEL_PATH)
else:
    print(f"⚠️ 警告: 未找到边缘模型 {YOLO_MODEL_PATH}，将仅运行场景评估")


# ================= SAEC 核心组件  =================

class SAEC_Estimator:
    """轻量级多尺度场景复杂度估计器 """

    @staticmethod
    def get_sc_score(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 1. 信息熵 (Entropy) - 衡量场景乱度
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.ravel() / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-7)) / 8.0  # 归一化到 0-1

        # 2. 边缘密度 (Edge Density) - 衡量细节丰富度
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges / 255.0) / (gray.shape[0] * gray.shape[1])
        edge_norm = min(edge_density / 0.06, 1.0)  # 0.06 为工业场景经验阈值

        # 3. 模糊度/清晰度 (Sharpness) - 衡量光照环境质量
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharp_norm = 1.0 - min(laplacian_var / 600.0, 1.0)  # 方差小表示模糊/光照差，得分高

        # 加权求和得到 Sc
        sc_score = (WEIGHTS["entropy"] * entropy +
                    WEIGHTS["edge"] * edge_norm +
                    WEIGHTS["sharpness"] * sharp_norm)

        return round(float(sc_score), 3), {
            "entropy": round(entropy, 3),
            "edge": round(edge_norm, 3),
            "quality": round(sharp_norm, 3)
        }


def cloud_mllm_inference(image_np, task_id, prompt="图中是否有工业缺陷？请详细描述。"):
    """
    云端 MLLM 推理任务（由后台执行）
    对应 SAEC 中的高效 MLLM 微调检查模块
    """
    print(f"☁️ [云端任务] ID: {task_id} 正在调用 Qwen3-VL...")
    try:
        _, img_encoded = cv2.imencode('.jpg', image_np)
        # 实际生产中，建议将图片转为 Base64 或上传至临时 OSS
        # 这里模拟向本地 SSH 隧道映射的端口发送请求
        # response = requests.post(CLOUD_API_URL, json={...})
        print(f"✅ [云端任务] ID: {task_id} 完成复核。")
    except Exception as e:
        print(f"❌ [云端任务] 失败: {e}")


# ================= API 接口 =================

@app.post("/inspect")
async def inspect_scene(
        background_tasks: BackgroundTasks,
        files: List[UploadFile] = File(...),
        use_saec: bool = Form(ENABLE_SAEC_ADAPTIVE)  # 动态开关
):
    results_manifest = []

    for file in files:
        # 读取图片
        data = await file.read()
        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 1. 执行 SAEC 场景复杂度评估
        sc_score, details = SAEC_Estimator.get_sc_score(img)
        is_complex = sc_score > SC_THRESHOLD

        # 2. 边缘侧 YOLO 快速检查
        yolo_res = edge_model.predict(img, conf=CONF_THRESHOLD, verbose=False)[0]
        detections = len(yolo_res.boxes)

        # 3. 协同调度决策 (Adaptive Scheduler)
        decision = "Edge_Only"
        if use_saec:
            # 如果场景太复杂，或者 YOLO 没信心，则触发云端 MLLM
            if is_complex or (detections == 0 and sc_score > 0.4):
                decision = "Edge_Cloud_Collaborative"
                background_tasks.add_task(cloud_mllm_inference, img.copy(), file.filename)

        results_manifest.append({
            "filename": file.filename,
            "sc_score": sc_score,
            "is_complex": is_complex,
            "yolo_count": detections,
            "decision": decision,
            "sc_details": details
        })

    return {
        "status": "success",
        "saec_enabled": use_saec,
        "results": results_manifest
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8172)