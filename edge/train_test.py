import torch
from ultralytics import YOLO


def train_best_strategy(task_mode='rgb'):
    """
    task_mode: 'rgb' (包含漏油、锈蚀等) 或 'heatmap' (包含过热、故障等)
    """
    # 加载预训练权重
    model = YOLO("yolov10m.pt")

    # 根据任务模式定制参数
    if task_mode == 'rgb':
        # 针对新增数百张漏油数据的 RGB 优化版
        data_path = r"D:\python_projects\dpsk-test\edge\datasets\yolo_rgb_dataset\data.yaml"
        special_params = {
            "batch": 16,  # 数据多了，Batch Size 调大能显著稳定梯度
            "copy_paste": 0.3,  # 略微降低，因为新数据多了，不需要像以前那样疯狂“粘贴”
            "mixup": 0.1,  # 保持较低，确保漏油边缘特征不模糊
            "cls": 1.0  # 标准分类权重
        }
    else:
        # 针对热力图（通常样本仍较少）的优化版
        data_path = r"D:\datasets\transformer_heatmap\data.yaml"
        special_params = {
            "batch": 8,  # 热力图数据量通常较小，保持较小 Batch
            "copy_paste": 0.1,
            "mixup": 0.2,  # 热力图可以多点 Mixup
            "cls": 2.0  # 提高分类损失，强迫模型关注稀少的热力故障类
        }

    # 最终最合适的方案配置
    model.train(
        data=data_path,
        project=f"transformer_{task_mode}",
        name="v2_data_boosted",
        epochs=300,
        imgsz=640,
        device=0,
        workers=0,  # Windows 稳定首选

        # --- 之前的核心有效参数恢复 ---
        box=7.5,  # 保持框的精准度
        overlap_mask=True,
        hsv_h=0.015,  # 色调微调
        hsv_s=0.7,  # 饱和度增强
        hsv_v=0.4,  # 亮度波动
        degrees=10.0,  # 轻微旋转
        fliplr=0.5,  # 水平翻转

        # --- 策略优化 ---
        mosaic=1.0,  # 必须开启，利用新数据增加小目标曝光
        close_mosaic=50,  # 增加到50轮，利用新增加的数据量进行最后的精度冲刺
        patience=80,  # 更有耐心，等待弱势类别收敛
        save=True,

        **special_params  # 载入差异化参数
    )


if __name__ == "__main__":
    # 建议先跑 RGB，因为你刚拿到了几百张新图
    train_best_strategy(task_mode='rgb')