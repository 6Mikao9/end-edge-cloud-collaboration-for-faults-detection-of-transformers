from ultralytics import YOLO

# 加载模型（首次会自动下载 yolov10n.pt）
model = YOLO(r"D:\python_projects\dpsk-test\edge\runs\detect\transformer_rgb\v2_data_boosted4\weights\best.pt")
#model = YOLO("yolov10n.pt")

# 推理
results = model.predict(source=r"D:\python_projects\dpsk-test\edge\datasets\yolo_rgb_dataset\val\images\194623_91c1dbc0-fe54-4f87-bc42-521fd6cd415c_9fcaa297-c72c-432e-aa4d-05e10493482d_jpg.rf.4e931c81dec23ad5e02bce3dbd570b48.jpg", save=True, show=True,conf=0.01)
#results = model.predict(source="D:\\python_projects\\dpsk-test\\edge\\datasets\\Transformer Fire Detection.v1i.yolov9\\train\images\\-90-Centerville-area-closed-off-due-to-transformer-fire-YouTube-Brave-2025-04-16-15-36-19_mp4-0000_jpg.rf.bb815a8e9982dc84f483bde2f2f8648c.jpg", save=True, show=True)