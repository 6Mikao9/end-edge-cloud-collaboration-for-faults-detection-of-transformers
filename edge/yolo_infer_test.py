from ultralytics import YOLO

# 加载模型（首次会自动下载 yolov10n.pt）
model = YOLO(r"D:\python_projects\dpsk-test\edge\runs\detect\transformer_project\yolov10_optimized7\weights\best.pt")
#model = YOLO("yolov10n.pt")

# 推理
results = model.predict(source=r"C:\Users\mille\Pictures\Screenshots\屏幕截图 2026-02-02 220229.png", save=True, show=True)
#results = model.predict(source="D:\\python_projects\\dpsk-test\\edge\\datasets\\Transformer Fire Detection.v1i.yolov9\\train\images\\-90-Centerville-area-closed-off-due-to-transformer-fire-YouTube-Brave-2025-04-16-15-36-19_mp4-0000_jpg.rf.bb815a8e9982dc84f483bde2f2f8648c.jpg", save=True, show=True)