import cv2
import time
import os
import numpy as np
import config


class EndSideSimulator:
    """
    边端模拟器类，整合视频处理、火灾检测、动作识别等功能
    """

    def __init__(self, video_path=None, debug_output_dir='debug_out', 
                 fire_brightness_threshold=None, fire_pixel_ratio=None,
                 motion_diff_threshold=None, motion_area_ratio=None,
                 gaussian_blur_size=None, sync_interval=None,
                 post_trigger_frames=None, frame_skip=None):
        """
        初始化边端模拟器
        
        Args:
            video_path: 视频文件路径，若为None则使用config.VIDEO_PATH
            debug_output_dir: 调试输出目录
            fire_brightness_threshold: 火光亮度阈值
            fire_pixel_ratio: 火光像素比例阈值
            motion_diff_threshold: 帧差阈值
            motion_area_ratio: 动作区域比例阈值
            gaussian_blur_size: 高斯模糊核大小
            sync_interval: 定时同步间隔（秒）
            post_trigger_frames: 触发后采样帧数
            frame_skip: 帧跳过数
        """
        self.video_path = video_path or config.VIDEO_PATH
        self.debug_output_dir = debug_output_dir
        
        # 火灾检测参数
        self.fire_brightness_threshold = fire_brightness_threshold or config.FIRE_BRIGHTNESS_THRESHOLD
        self.fire_pixel_ratio = fire_pixel_ratio or config.FIRE_PIXEL_RATIO
        
        # 动作检测参数
        self.motion_diff_threshold = motion_diff_threshold or config.MOTION_DIFF_THRESHOLD
        self.motion_area_ratio = motion_area_ratio or config.MOTION_AREA_RATIO
        self.gaussian_blur_size = gaussian_blur_size or config.GAUSSIAN_BLUR_SIZE
        
        # 采样参数
        self.sync_interval = sync_interval or config.SYNC_INTERVAL
        self.post_trigger_frames = post_trigger_frames or config.POST_TRIGGER_FRAMES
        self.frame_skip = frame_skip or config.FRAME_SKIP
        
        # 内部状态
        self.cap = None
        self.last_gray = None
        self.last_sync_time = None
        self.is_running = False
        
        self._ensure_output_dir()

    def _ensure_output_dir(self):
        """确保输出目录存在"""
        if not os.path.exists(self.debug_output_dir):
            os.makedirs(self.debug_output_dir)

    def send_to_edge(self, frames, reason):
        """
        保存采样帧到输出目录
        命名逻辑：时间_原因_序号
        
        Args:
            frames: 帧列表
            reason: 触发原因
        """
        # 获取格式化时间字符串
        time_str = time.strftime("%Y%m%d_%H%M%S")

        print(f"\n [发送中] 原因: {reason} | 采样时间: {time_str}")

        for i, f in enumerate(frames):
            filename = os.path.join(self.debug_output_dir, f"{time_str}_{reason}_{i}.jpg")
            cv2.imwrite(filename, f)
            print(f"   -> 已保存: {filename}")

    def detect_fire(self, frame, gray):
        """
        火灾检测（双重判定：亮度+颜色）
        
        Args:
            frame: BGR帧
            gray: 灰度帧
            
        Returns:
            bool: 是否检测到火灾
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # 定义火光的大致颜色范围（红色、橙色、黄色区域）
        lower_fire = np.array([0, 100, 200])
        upper_fire = np.array([35, 255, 255])
        mask = cv2.inRange(hsv, lower_fire, upper_fire)
        fire_pixel_count = np.sum(mask > 0)
        fire_ratio = fire_pixel_count / mask.size

        return fire_ratio > self.fire_pixel_ratio or np.mean(gray) > self.fire_brightness_threshold

    def detect_motion(self, gray):
        """
        帧差法动作检测（增加形态学处理）
        
        Args:
            gray: 灰度帧
            
        Returns:
            bool: 是否检测到动作
        """
        if self.last_gray is None:
            return False

        diff = cv2.absdiff(gray, self.last_gray)
        _, thresh = cv2.threshold(diff, self.motion_diff_threshold, 255, cv2.THRESH_BINARY)
        # 使用膨胀操作，合并小的噪点
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)

        change_ratio = np.sum(thresh == 255) / thresh.size
        return change_ratio > self.motion_area_ratio

    def check_timer_sync(self):
        """
        检查是否到达定时同步时间
        
        Returns:
            bool: 是否需要定时同步
        """
        return time.time() - self.last_sync_time > self.sync_interval

    def process_frame(self, frame):
        """
        处理单帧，检测触发条件
        
        Args:
            frame: 原始帧
            
        Returns:
            tuple: (trigger_reason, processed_frame) 触发原因和处理后的帧
        """
        small_frame = cv2.resize(frame, (640, 480))
        # 增加高斯模糊，显著降低由于噪声引起的"敏感度"
        blurred = cv2.GaussianBlur(small_frame, 
                                    (self.gaussian_blur_size, self.gaussian_blur_size), 0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

        trigger_reason = None

        # 火灾检测
        if self.detect_fire(small_frame, gray):
            trigger_reason = "Fire_Alarm"

        # 帧差法动作检测
        elif self.detect_motion(gray):
            trigger_reason = "Motion_Change"

        # 定时逻辑
        elif self.check_timer_sync():
            trigger_reason = "Timer_Sync"

        self.last_gray = gray
        return trigger_reason, small_frame

    def capture_frames(self, count):
        """
        从视频中捕获指定数量的帧
        
        Args:
            count: 要捕获的帧数
            
        Returns:
            list: 捕获的帧列表
        """
        frames = []
        for _ in range(count - 1):
            for _ in range(self.frame_skip):
                self.cap.read()
            success, frame = self.cap.read()
            if success:
                frames.append(frame)
            else:
                break
        return frames

    def run(self, show_preview=True):
        """
        运行边端模拟器主循环
        
        Args:
            show_preview: 是否显示预览窗口
        """
        self.cap = cv2.VideoCapture(self.video_path)
        self.last_gray = None
        self.last_sync_time = time.time()
        self.is_running = True

        try:
            while self.cap.isOpened() and self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    break

                trigger_reason, small_frame = self.process_frame(frame)

                # 执行触发采样
                if trigger_reason:
                    frames_to_send = [frame] + self.capture_frames(self.post_trigger_frames)
                    self.send_to_edge(frames_to_send, trigger_reason)
                    self.last_sync_time = time.time()  # 无论什么原因上报，都重置定时器

                # 预览与退出
                if show_preview:
                    cv2.imshow('End_Side', small_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        finally:
            self.stop()

    def stop(self):
        """停止运行，释放资源"""
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

    def __del__(self):
        """析构函数，确保资源释放"""
        self.stop()


def main():
    """测试函数，保留原有测试功能"""
    simulator = EndSideSimulator()
    simulator.run(show_preview=True)


if __name__ == "__main__":
    main()
