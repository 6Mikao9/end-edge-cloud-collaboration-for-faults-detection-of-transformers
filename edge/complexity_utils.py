import cv2
import numpy as np


class SceneComplexityEstimator:
    def __init__(self, entropy_weight=0.4, edge_weight=0.3, sharp_weight=0.3):
        # SAEC 逻辑：通过加权三个指标来评估场景复杂度
        self.w1 = entropy_weight
        self.w2 = edge_weight
        self.w3 = sharp_weight

    def get_entropy(self, image):
        """计算图像信息熵：反映背景杂乱度"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.ravel() / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-7))
        # 归一化处理（通常 0-8 之间）
        return min(entropy / 8.0, 1.0)

    def get_edge_density(self, image):
        """计算边缘密度：反映物体丰富度 """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        density = np.sum(edges / 255.0) / (image.shape[0] * image.shape[1])
        # 归一化（根据经验，0.05以上就很复杂了）
        return min(density / 0.05, 1.0)

    def get_sharpness(self, image):
        """计算拉普拉斯方差：反映光照与清晰度"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        # 归一化：方差越小越模糊，1-归一化值代表“低质量”程度
        norm_sharp = min(variance / 500.0, 1.0)
        return 1.0 - norm_sharp

    def calculate_score(self, image):
        """计算最终的 Sc 得分"""
        s1 = self.get_entropy(image)
        s2 = self.get_edge_density(image)
        s3 = self.get_sharpness(image)

        # Sc = w1*Entropy + w2*Edge + w3*Low_Sharpness
        sc_score = (self.w1 * s1) + (self.w2 * s2) + (self.w3 * s3)
        return sc_score, {"entropy": s1, "edge": s2, "low_sharpness": s3}