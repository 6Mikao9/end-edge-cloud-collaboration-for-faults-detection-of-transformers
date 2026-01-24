from ultralytics import YOLO
import torch


def train_model():
    # 1. 加载预训练权重 (yolov10n 是最小的模型，训练压力小)
    # 如果你本地没有这个文件，程序会自动下载
    model = YOLO("yolov10n.pt")

    # 2. 开始训练
    # data: 指向你解压出来的 data.yaml
    # epochs: 训练轮数。新手建议先跑 50-100 轮
    # imgsz: 图片尺寸。变压器检测建议用 640
    # batch: 每一批处理的图片数。显存小就调小（如 8 或 16）
    # device: 0 代表使用第一张显卡，如果没有显卡请写 'cpu'
    results = model.train(
        data="D:\\python_projects\\dpsk-test\\edge\\datasets\\Transformer Fire Detection.v1i.yolov9\\data.yaml",  # 👈 换成你 data.yaml 的绝对路径
        epochs=300,
        imgsz=640,
        batch=48,
        freeze=10,
        device=0,
        workers=4,
        project="transformer_project",  # 结果保存的文件夹名
        name="yolov10_test"  # 实验名
    )

    print("训练完成！结果保存在 runs/detect/yolov10_test 文件夹下")


if __name__ == "__main__":
    # 检查 GPU 是否可用
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    train_model()