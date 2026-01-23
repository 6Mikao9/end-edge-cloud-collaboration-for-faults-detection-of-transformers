import cv2
import time
import os
import numpy as np
import config


def send_to_edge(frames, reason):
    """
    修改后的命名逻辑：时间_原因_序号
    """
    if not os.path.exists('debug_out'):
        os.makedirs('debug_out')

    # 获取格式化时间字符串
    time_str = time.strftime("%Y%m%d_%H%M%S")

    print(f"\n🚀 [发送中] 原因: {reason} | 采样时间: {time_str}")

    for i, f in enumerate(frames):
        filename = f"debug_out/{time_str}_{reason}_{i}.jpg"
        cv2.imwrite(filename, f)
        print(f"   -> 已保存: {filename}")


def main():
    cap = cv2.VideoCapture(config.VIDEO_PATH)
    last_gray = None
    last_sync_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        small_frame = cv2.resize(frame, (640, 480))
        # 增加高斯模糊，显著降低由于噪声引起的“敏感度”
        blurred = cv2.GaussianBlur(small_frame, (config.GAUSSIAN_BLUR_SIZE, config.GAUSSIAN_BLUR_SIZE), 0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

        trigger_reason = None

        # --- 优化后的起火检测 (双重判定：亮度+颜色) ---
        hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
        # 定义火光的大致颜色范围（红色、橙色、黄色区域）
        lower_fire = np.array([0, 100, 200])
        upper_fire = np.array([35, 255, 255])
        mask = cv2.inRange(hsv, lower_fire, upper_fire)
        fire_pixel_count = np.sum(mask > 0)
        fire_ratio = fire_pixel_count / mask.size

        if fire_ratio > config.FIRE_PIXEL_RATIO or np.mean(gray) > config.FIRE_BRIGHTNESS_THRESHOLD:
            trigger_reason = "Fire_Alarm"

        # --- 优化后的帧差法 (增加形态学处理) ---
        if not trigger_reason and last_gray is not None:
            diff = cv2.absdiff(gray, last_gray)
            _, thresh = cv2.threshold(diff, config.MOTION_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
            # 使用膨胀操作，合并小的噪点，过滤由于摄像头抖动产生的细碎边缘
            kernel = np.ones((5, 5), np.uint8)
            thresh = cv2.dilate(thresh, kernel, iterations=1)

            change_ratio = np.sum(thresh == 255) / thresh.size
            if change_ratio > config.MOTION_AREA_RATIO:
                trigger_reason = "Motion_Change"

        # --- 定时逻辑 ---
        if not trigger_reason:
            if time.time() - last_sync_time > config.SYNC_INTERVAL:
                trigger_reason = "Timer_Sync"

        # --- 执行触发采样 ---
        if trigger_reason:
            frames_to_send = [frame]
            for _ in range(config.POST_TRIGGER_FRAMES - 1):
                for _ in range(config.FRAME_SKIP): cap.read()
                success, next_f = cap.read()
                if success: frames_to_send.append(next_f)

            send_to_edge(frames_to_send, trigger_reason)
            last_sync_time = time.time()  # 无论什么原因上报，都重置定时器

        # 预览与退出
        cv2.imshow('End_Side', small_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        last_gray = gray

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()