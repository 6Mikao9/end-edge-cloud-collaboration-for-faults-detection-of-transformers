import os
import cv2
import requests
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from typing import List
from ultralytics import YOLO

# ================= 配置区 =================
MODEL_PATH = r"D:\python_projects\dpsk-test\edge\runs\detect\transformer_project\yolov10_test6\weights\best.pt"
CLOUD_URL = "http://127.0.0.1:8972/vlm_inference"
CONF_THRESHOLD = 0.4

app = FastAPI()

if not os.path.exists(MODEL_PATH):
    print(f"❌ 致命错误：找不到模型文件 -> {MODEL_PATH}")
    exit(1)
model = YOLO(MODEL_PATH)


def crop_box(image, box, expand=0.1):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    ew, eh = (x2 - x1) * expand, (y2 - y1) * expand
    x1, y1 = max(0, int(x1 - ew)), max(0, int(y1 - eh))
    x2, y2 = min(w, int(x2 + ew)), min(h, int(y2 + eh))
    return image[y1:y2, x1:x2]


# --- 核心修改：这是将在后台运行的函数 ---
def background_cloud_task(image_np, reason, mode):
    """
    这个函数会在给端侧返回响应后，在后台悄悄运行
    """
    print(f"🔄 [后台任务启动] 正在将 {mode} 图片发往云端...")
    try:
        success, encoded_img = cv2.imencode('.jpg', image_np)
        if not success: return

        files = {'file': (f'{mode}_{reason}.jpg', encoded_img.tobytes(), 'image/jpeg')}
        data = {'reason': reason, 'mode': mode}

        # 这里依然会阻塞，但只阻塞后台线程，不影响前台响应
        resp = requests.post(CLOUD_URL, files=files, data=data, timeout=60)
        print(f"✅ [后台任务完成] 云端已接收并处理: {resp.status_code}")
        # 注意：这里拿到的结果没法直接返给端侧了（因为端侧早走了）
        # 通常做法是写入日志或数据库

    except Exception as e:
        print(f"❌ [后台任务失败] 连接云端出错: {e}")


@app.post("/predict")
async def predict(
        background_tasks: BackgroundTasks,  # 👈 注入后台任务管理器
        reason: str = Form(...),
        files: List[UploadFile] = File(...)
):
    print(f"\n📥 [边缘收到] 原因: {reason} | 图片数: {len(files)}")

    images = []
    for file in files:
        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None: images.append(img)

    if not images: return {"error": "No valid images"}

    # YOLO 推理 (这个很快，几百毫秒，可以同步做)
    results = model.predict(images, conf=CONF_THRESHOLD, verbose=False)

    # --- 决策逻辑 (由同步改为添加后台任务) ---
    triggered_tasks = 0
    for i, res in enumerate(results):
        boxes = res.boxes

        # 情况A: 漏检复核
        if "Fire" in reason and len(boxes) == 0:
            print("   -> 计划任务: 发送整图至云端")
            # 关键：不要直接调用，而是 add_task
            background_tasks.add_task(background_cloud_task, images[i].copy(), "Suspected_Fire_Missed", "full")
            triggered_tasks += 1

        # 情况B: 细节复核
        for box in boxes:
            cls_name = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            if ('fire' in cls_name.lower() or 'smoke' in cls_name.lower()):
                print(f"   -> 计划任务: 发送 {cls_name} 局部至云端")
                crop_img = crop_box(images[i], box)
                # 关键：发送副本，防止内存被释放
                background_tasks.add_task(background_cloud_task, crop_img.copy(), f"Check_{cls_name}", "crop")
                triggered_tasks += 1

    # 🚀 立即返回，不等待云端结果
    return {
        "status": "accepted",
        "message": "Edge processing done, cloud tasks queued.",
        "yolo_detections": len(results[0].boxes),
        "queued_cloud_tasks": triggered_tasks
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8172)