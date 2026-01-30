import cv2
import time
import os
import sys
import uuid
import numpy as np
import threading
import requests
import json
from datetime import datetime
import config


class EndSideSimulator:
    """
    端侧设备模拟器
    - 设备ID：随机生成且持久化保存
    - 心跳机制：定期向边缘服务器报告存活状态
    - 故障检测：自动检测并上报设备异常
    """
    
    DEVICE_ID_FILE = "device_id.txt"
    
    def __init__(self, video_path=None, debug_output_dir='debug_out',
                 inference_interval=2.0,
                 fire_brightness_threshold=None, fire_pixel_ratio=None,
                 motion_diff_threshold=None, motion_area_ratio=None,
                 gaussian_blur_size=None, sync_interval=None,
                 post_trigger_frames=None, frame_skip=None,
                 edge_url=None, heartbeat_interval=30):

        self.video_path = video_path or config.VIDEO_PATH
        self.debug_output_dir = debug_output_dir
        self.edge_url = edge_url or getattr(config, 'EDGE_SERVER_URL', "http://127.0.0.1:8172")
        self.edge_predict_url = f"{self.edge_url}/predict"
        self.edge_heartbeat_url = f"{self.edge_url}/heartbeat"

        self.inference_interval = inference_interval
        self.last_inference_time = 0

        self.fire_brightness_threshold = fire_brightness_threshold or config.FIRE_BRIGHTNESS_THRESHOLD
        self.fire_pixel_ratio = fire_pixel_ratio or config.FIRE_PIXEL_RATIO
        self.motion_diff_threshold = motion_diff_threshold or config.MOTION_DIFF_THRESHOLD
        self.motion_area_ratio = motion_area_ratio or config.MOTION_AREA_RATIO
        self.gaussian_blur_size = gaussian_blur_size or config.GAUSSIAN_BLUR_SIZE

        self.sync_interval = sync_interval or config.SYNC_INTERVAL
        self.post_trigger_frames = post_trigger_frames or config.POST_TRIGGER_FRAMES
        self.frame_skip = frame_skip or config.FRAME_SKIP

        self.cap = None
        self.last_gray = None
        self.last_sync_time = time.time()
        self.is_running = False
        
        # 设备ID管理
        self.device_id = self._get_or_create_device_id()
        self.heartbeat_interval = heartbeat_interval
        self.last_heartbeat_time = 0
        self.device_status = "normal"  # normal, warning, error
        self.error_message = ""

        self._ensure_output_dir()
        print(f"🎥 端侧设备启动 | 设备ID: {self.device_id}")

    def _get_or_create_device_id(self):
        """获取或创建设备ID（持久化存储）"""
        # 首先尝试从文件读取
        if os.path.exists(self.DEVICE_ID_FILE):
            try:
                with open(self.DEVICE_ID_FILE, 'r') as f:
                    device_id = f.read().strip()
                    if device_id:
                        return device_id
            except Exception as e:
                print(f"⚠️ 读取设备ID文件失败: {e}")
        
        # 生成新的设备ID
        # 格式: END-{MAC后缀}-{随机UUID前8位}
        mac_part = uuid.getnode() % 10000
        uuid_part = str(uuid.uuid4())[:8].upper()
        device_id = f"END-{mac_part:04d}-{uuid_part}"
        
        # 保存到文件
        try:
            with open(self.DEVICE_ID_FILE, 'w') as f:
                f.write(device_id)
        except Exception as e:
            print(f"⚠️ 保存设备ID文件失败: {e}")
        
        return device_id

    def _ensure_output_dir(self):
        if not os.path.exists(self.debug_output_dir):
            os.makedirs(self.debug_output_dir)

    def _http_upload_worker(self, frames_copy, reason):
        """HTTP上传工作线程"""
        files = []
        for i, frame in enumerate(frames_copy):
            success, buffer = cv2.imencode('.jpg', frame)
            if success:
                files.append(('files', (f'frame_{i}.jpg', buffer.tobytes(), 'image/jpeg')))
        try:
            payload = {
                'reason': reason, 
                'timestamp': time.time(),
                'device_id': self.device_id  # 携带设备ID
            }
            response = requests.post(self.edge_predict_url, files=files, data=payload, timeout=10)
            print(f"   -> [HTTP后台] 异步发送成功 | 设备: {self.device_id}")
        except Exception as e:
            print(f"   -> [HTTP后台] 发送失败: {e}")
            self.device_status = "warning"
            self.error_message = f"Upload failed: {str(e)}"

    def _async_upload(self, frames, reason):
        frames_copy = [f.copy() for f in frames]
        thread = threading.Thread(target=self._http_upload_worker, args=(frames_copy, reason))
        thread.daemon = True
        thread.start()

    def send_to_edge(self, frames, reason):
        time_str = time.strftime("%Y%m%d_%H%M%S")
        print(f"\n🚀 [检测触发] 原因: {reason} | 时间: {time_str} | 设备: {self.device_id}")
        for i, f in enumerate(frames):
            filename = os.path.join(self.debug_output_dir, f"{time_str}_{reason}_{i}.jpg")
            cv2.imwrite(filename, f)
        self._async_upload(frames, reason)

    def detect_fire(self, frame, gray):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 100, 200]), np.array([35, 255, 255]))
        fire_ratio = np.sum(mask > 0) / mask.size
        return fire_ratio > self.fire_pixel_ratio or np.mean(gray) > self.fire_brightness_threshold

    def detect_motion(self, gray):
        if self.last_gray is None: return False
        diff = cv2.absdiff(gray, self.last_gray)
        _, thresh = cv2.threshold(diff, self.motion_diff_threshold, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, np.ones((5, 5), np.uint8), iterations=1)
        return (np.sum(thresh == 255) / thresh.size) > self.motion_area_ratio

    def send_heartbeat(self):
        """发送心跳包到边缘服务器"""
        try:
            payload = {
                'device_id': self.device_id,
                'timestamp': time.time(),
                'status': self.device_status,
                'error_message': self.error_message if self.device_status != "normal" else "",
                'video_source': self.video_path,
                'inference_interval': self.inference_interval
            }
            response = requests.post(
                self.edge_heartbeat_url, 
                json=payload, 
                timeout=5
            )
            if response.status_code == 200:
                # 清除已上报的错误状态
                if self.device_status == "warning":
                    self.device_status = "normal"
                    self.error_message = ""
        except Exception as e:
            print(f"⚠️ 心跳发送失败: {e}")

    def process_frame(self, frame):
        small_frame = cv2.resize(frame, (640, 480))
        blurred = cv2.GaussianBlur(small_frame, (self.gaussian_blur_size, self.gaussian_blur_size), 0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

        trigger_reason = None
        if self.detect_fire(small_frame, gray):
            trigger_reason = "Fire_Alarm"
        elif self.detect_motion(gray):
            trigger_reason = "Motion_Change"
        elif (time.time() - self.last_sync_time > self.sync_interval):
            trigger_reason = "Timer_Sync"

        self.last_gray = gray
        return trigger_reason, small_frame

    def capture_frames(self, count):
        frames = []
        for _ in range(count - 1):
            for _ in range(self.frame_skip): self.cap.read()
            success, frame = self.cap.read()
            if success: frames.append(frame)
        return frames

    def run(self, show_preview=True):
        self.cap = cv2.VideoCapture(self.video_path)
        
        if not self.cap.isOpened():
            print(f"❌ 无法打开视频源: {self.video_path}")
            self.device_status = "error"
            self.error_message = f"Cannot open video source: {self.video_path}"
            return

        video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0: video_fps = 25
        wait_ms = int(1000 / video_fps)
        print(f"🎬 视频原生频率: {video_fps} FPS | 算法推理间隔: {self.inference_interval}s")
        print(f"💓 心跳间隔: {self.heartbeat_interval}s")

        self.last_sync_time = time.time()
        self.is_running = True
        frame_count = 0

        try:
            while self.cap.isOpened() and self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    # 视频循环播放
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                frame_count += 1
                current_time = time.time()
                
                # 发送心跳（定期）
                if current_time - self.last_heartbeat_time >= self.heartbeat_interval:
                    self.send_heartbeat()
                    self.last_heartbeat_time = current_time

                # 只有到达采样间隔才推理
                if current_time - self.last_inference_time >= self.inference_interval:
                    trigger_reason, _ = self.process_frame(frame)
                    self.last_inference_time = current_time
                    if trigger_reason:
                        frames_to_send = [frame] + self.capture_frames(self.post_trigger_frames)
                        self.send_to_edge(frames_to_send, trigger_reason)
                        self.last_sync_time = current_time

                if show_preview:
                    preview_img = cv2.resize(frame, (640, 480))
                    # 显示设备ID和状态
                    status_color = (0, 255, 0) if self.device_status == "normal" else (0, 0, 255)
                    cv2.putText(preview_img, f"ID: {self.device_id}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
                    cv2.putText(preview_img, f"Status: {self.device_status}", (10, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
                    cv2.putText(preview_img, f"Inference: {self.inference_interval}s", (10, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    cv2.imshow('End_Side_Simulator', preview_img)

                    if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
                        break
        finally:
            self.stop()

    def stop(self):
        self.is_running = False
        # 发送离线心跳
        try:
            requests.post(
                self.edge_heartbeat_url,
                json={
                    'device_id': self.device_id,
                    'timestamp': time.time(),
                    'status': 'offline'
                },
                timeout=3
            )
        except:
            pass
        if self.cap: self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # 支持命令行参数指定设备ID前缀
    device_prefix = sys.argv[1] if len(sys.argv) > 1 else None
    
    simulator = EndSideSimulator(inference_interval=2.0)
    simulator.run(show_preview=True)
