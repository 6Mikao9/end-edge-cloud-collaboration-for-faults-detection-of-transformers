import os
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from http import HTTPStatus
import dashscope

# ================= 配置区 =================
dashscope.api_key = "sk-eb2d703196d3485a88ef34aa08d27c82"
TEMP_DIR = "static/uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

latest_status = {
    "is_fire": False,
    "reason": "等待检测...",
    "img_url": "",
    "detail": "无"
}


def call_qwen_vl_custom(image_path, prompt):
    """
    通用调用函数，接受自定义 prompt
    """
    local_file_uri = f"file://{os.path.abspath(image_path)}"

    # 在 prompt 后面强制加上格式要求
    full_prompt = f"{prompt}。请严格遵守回复格式：[有/无]|[10字内理由]"

    messages = [{"role": "user", "content": [{"image": local_file_uri}, {"text": full_prompt}]}]

    try:
        response = dashscope.MultiModalConversation.call(model='qwen-vl-max', messages=messages)
        if response.status_code == HTTPStatus.OK:
            return response.output.choices[0].message.content
        return "错误|API访问失败"
    except Exception as e:
        return f"错误|{str(e)}"


@app.post("/vlm_inference")
async def vlm_handler(reason: str = Form(...), mode: str = Form(...), file: UploadFile = File(...)):
    global latest_status

    # 1. 保存图片
    file_name = f"{mode}_{file.filename}"
    file_path = os.path.join(TEMP_DIR, file_name)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # 2. 根据上传原因设置针对性提示词
    # 逻辑：如果是火灾相关用一套，其他异常（如漏油、异物）用另一套
    if "FIRE" in reason.upper() or "SMOKE" in reason.upper():
        base_prompt = "图中是否有明火或浓烟？"
    elif "CHECK" in reason.upper():
        base_prompt = "图中变压器设备是否有破损、异物挂载或明显异常？"
    else:
        base_prompt = f"分析此变压器画面，是否存在{reason}提及的风险？"

    # 3. 调用大模型
    raw_response = call_qwen_vl_custom(file_path, base_prompt)
    print(f"🤖 大模型针对性输出 ({reason}): {raw_response}")

    # 4. 解析结果 (处理列表或字符串)
    if isinstance(raw_response, list) and len(raw_response) > 0:
        raw_result = raw_response[0].get('text', '')
    else:
        raw_result = str(raw_response)

    # 5. 更新全局状态
    # 只要结果里包含“有”，就触发网页报警
    is_fire_detected = "有" in raw_result.split('|')[0]
    detail_info = raw_result.split('|')[-1] if '|' in raw_result else raw_result

    latest_status = {
        "is_fire": is_fire_detected,
        "reason": reason,
        "img_url": f"/static/uploads/{file_name}",
        "detail": detail_info
    }

    return {"status": "success", "analysis": raw_result}


@app.get("/get_latest")
async def get_latest():
    return latest_status


@app.get("/", response_class=HTMLResponse)
async def index():
    if os.path.exists("static/index.html"):
        with open("static/index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Missing static/index.html</h1>"


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8972)