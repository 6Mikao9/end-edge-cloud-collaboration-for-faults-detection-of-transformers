import os
import cv2
import requests
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from typing import List, Optional
from ultralytics import YOLO

# ================= SAEC 配置区 =================
# 1. 边缘模型路径 (本地 Ubuntu 运行)
YOLO_MODEL_PATH = r"D:\python_projects\dpsk-test\edge\runs\detect\transformer_project\yolov10_test6\weights\best.pt"

# 2. SAEC 调度阈值与权重 (参考论文指标进行量化)
ENABLE_SAEC_ADAPTIVE = True  # 总开关：是否启用场景感知自适应调度
SC_THRESHOLD = 0.55  # 复杂度阈值：超过此值则认为场景复杂，需云端介入
WEIGHTS = {
    "entropy": 0.4,  # 信息熵权重：衡量背景杂乱度
    "edge": 0.3,  # 边缘密度权重：衡量物体细节丰富度
    "sharpness": 0.3  # 清晰度权重：衡量光照环境和模糊度
}

# 3. 业务参数
CONF_THRESHOLD = 0.45  # 边缘 YOLO 置信度

app = FastAPI(title="SAEC Scene-Aware Collaborative System")

# 初始化边缘模型
if os.path.exists(YOLO_MODEL_PATH):
    edge_model = YOLO(YOLO_MODEL_PATH)
else:
    print(f"⚠️ 警告: 未找到模型 {YOLO_MODEL_PATH}，系统将仅运行场景复杂度评估。")


# ================= SAEC 场景感知核心逻辑 =================

class SAEC_Estimator:
    """轻量级多尺度场景复杂度估计器"""

    @staticmethod
    def get_sc_score(image):
        """
        根据 SAEC 算法计算场景复杂度得分 Sc
        Sc 越高，代表环境越差（光照不足、背景杂乱、目标细小）
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 1. 计算信息熵 (Entropy) - 评估环境不确定性
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.ravel() / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-7)) / 8.0  # 归一化

        # 2. 计算边缘密度 (Edge Density) - 评估场景细节
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges / 255.0) / (gray.shape[0] * gray.shape[1])
        edge_norm = min(edge_density / 0.06, 1.0)  # 0.06 为典型工业背景经验值

        # 3. 计算清晰度 (Sharpness) - 评估光照质量
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharp_norm = 1.0 - min(laplacian_var / 600.0, 1.0)  # 方差小意味着模糊/暗光

        # 综合加权计算 Sc 得分
        sc_score = (WEIGHTS["entropy"] * entropy +
                    WEIGHTS["edge"] * edge_norm +
                    WEIGHTS["sharpness"] * sharp_norm)

        return round(float(sc_score), 3), {
            "entropy_score": round(entropy, 3),
            "edge_score": round(edge_norm, 3),
            "dim_blur_score": round(sharp_norm, 3)
        }


# ================= 业务处理接口 =================

@app.post("/inspect")
async def inspect(
        background_tasks: BackgroundTasks,
        files: List[UploadFile] = File(...),
        use_saec: bool = Form(ENABLE_SAEC_ADAPTIVE)
):
    """
    SAEC 协作巡检接口
    实现逻辑：
    1. 边缘端快速计算场景复杂度
    2. 如果场景简单且边缘模型检出，则本地直接处理
    3. 如果场景复杂，自动触发云端 MLLM 进行高级推理
    """
    results_list = []

    for file in files:
        # 读取并解码图像
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # --- [第一步] 场景感知评估 ---
        sc_score, sc_details = SAEC_Estimator.get_sc_score(img)
        is_complex_scene = sc_score > SC_THRESHOLD

        # --- [第二步] 边缘侧 YOLO 快速推理 ---
        detections = []
        if os.path.exists(YOLO_MODEL_PATH):
            yolo_results = edge_model.predict(img, conf=CONF_THRESHOLD, verbose=False)[0]
            for box in yolo_results.boxes:
                detections.append({
                    "class": edge_model.names[int(box.cls[0])],
                    "conf": float(box.conf[0])
                })

        # --- [第三步] 自适应协同调度决策 ---
        # 决策逻辑：
        # 1. 开启了 SAEC 开关
        # 2. 场景复杂度 Sc 超过阈值，认为边缘模型不可靠
        # 3. 或者 YOLO 漏检但环境处于疑似复杂状态 (Sc > 0.4)
        trigger_cloud = False
        if use_saec:
            if is_complex_scene or (len(detections) == 0 and sc_score > 0.4):
                trigger_cloud = True
                # 这里放入后台任务，实际调用你微调好的 Qwen3-VL
                # background_tasks.add_task(call_cloud_vlm, img.copy(), file.filename)

        results_list.append({
            "filename": file.filename,
            "complexity_score": sc_score,
            "scene_status": "Complex/Low-Quality" if is_complex_scene else "Normal/Simple",
            "edge_detections": detections,
            "collaborative_decision": "Trigger_Cloud_MLLM" if trigger_cloud else "Local_Edge_Only",
            "sc_metrics": sc_details
        })

    return {
        "status": "success",
        "saec_config": {"threshold": SC_THRESHOLD, "enabled": use_saec},
        "results": results_list
    }


if __name__ == "__main__":
    # 建议在 Ubuntu 小服务器上运行此端口
    uvicorn.run(app, host="0.0.0.0", port=8172)