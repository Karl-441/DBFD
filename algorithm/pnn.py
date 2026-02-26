import numpy as np
from scipy.spatial.distance import cdist
import config

"""
PNN 概率神经网络模块 (Inference Only)
    实现了一个概率神经网络 (PNN) 分类器。
    仅保留预测功能，移除训练相关代码。
"""

class PNN:
    def __init__(self):
        self.X_train = None  # 训练样本特征 (Loaded from pickle)
        self.y_train = None  # 训练样本标签 (Loaded from pickle)
        self.sigmas = {}     # 类别 -> 平滑因子 (Sigma) 的映射 (Loaded from pickle)
        self.classes = []    # 类别列表
        
    def predict(self, X):
        """
        预测新样本,对新样本进行分类
        参数:
            X (array-like): 测试样本特征
            
        返回值:
            numpy.ndarray: 预测标签数组
        """
        X = np.array(X)
        if X.ndim == 1:
            X = X[None, :]
            
        n_test = len(X)
        dists = cdist(X, self.X_train, 'sqeuclidean')
        
        preds = []
        dim = self.X_train.shape[1]
        
        for i in range(n_test):
            scores = {}
            for c in self.classes:
                sigma = self.sigmas[c]
                c_indices = np.where(self.y_train == c)[0]
                
                if len(c_indices) == 0:
                    scores[c] = 0
                    continue
                
                d = dists[i, c_indices]
                norm = (2 * np.pi * sigma**2) ** (dim / 2)
                kernels = np.exp(-d / (2 * sigma**2))
                density = np.sum(kernels) / (len(c_indices) * norm)
                scores[c] = density
            
            pred = max(scores, key=scores.get)
            preds.append(pred)
            
        return np.array(preds)
