import numpy as np
from scipy.spatial.distance import cdist
try:
    from scipy.cluster.vq import kmeans2
except ImportError:
    kmeans2 = None
import config

"""
PNN 概率神经网络模块
    实现了一个概率神经网络 (PNN) 分类器。
    PNN 基于贝叶斯决策规则和 Parzen 窗概率密度估计。
    本实现包含了一个基于 ECM (Expectation-Conditional Maximization) 思想的
    平滑因子 (Sigma) 优化算法，用于自动调整每个类别的最佳 Sigma 值。
"""

class PNN:
    def __init__(self):
        self.X_train = None  # 训练样本特征
        self.y_train = None  # 训练样本标签
        self.sigmas = {}     # 类别 -> 平滑因子 (Sigma) 的映射
        self.classes = []    # 类别列表
        
    def fit(self, X, y):
        """
        训练模型
            存储训练数据并初始化平滑因子。
            如果数据量超过配置限制，将使用 K-Means 聚类算法压缩模式层，减少内存占用并提高预测速度。
            
        参数:
            X (array-like): 训练特征矩阵
            y (array-like): 训练标签
        """
        X = np.array(X)
        y = np.array(y)
        self.classes = np.unique(y)
        
        # 模式压缩 (Pattern Compression)
        if kmeans2 and config.PNN_MAX_SAMPLES > 0:
            X_compressed = []
            y_compressed = []
            
            for c in self.classes:
                # 获取该类的所有样本
                c_indices = np.where(y == c)[0]
                X_c = X[c_indices]
                
                # 如果样本数超过限制，进行聚类
                if len(X_c) > config.PNN_MAX_SAMPLES:
                    print(f"PNN: Compressing class {c} from {len(X_c)} to {config.PNN_MAX_SAMPLES} prototypes...")
                    # 使用 K-Means 聚类，centroids 作为新的原型向量
                    # minit='points' 选择初始中心，iter=10 迭代次数
                    centroids, _ = kmeans2(X_c, k=config.PNN_MAX_SAMPLES, minit='points', iter=10)
                    X_compressed.extend(centroids)
                    y_compressed.extend([c] * len(centroids))
                else:
                    X_compressed.extend(X_c)
                    y_compressed.extend([c] * len(X_c))
            
            self.X_train = np.array(X_compressed)
            self.y_train = np.array(y_compressed)
        else:
            self.X_train = X
            self.y_train = y
            
        # 初始化所有类别的 sigma 为 1.0
        for c in self.classes:
            self.sigmas[c] = 1.0

    def optimize_ecm(self):
        """
        优化平滑因子
            使用类似 ECM (期望条件最大化) 的迭代方法优化每个类别的 Sigma 值。
            目标是最大化留一法 (Leave-One-Out) 的准确率/似然度。
            采用坐标下降法，依次优化每个类别的 Sigma。
        """
        print("Starting ECM optimization for PNN smoothing factors... (开始优化 PNN 平滑因子)")
        
        # 迭代优化每个 sigma
        max_iter = 5
        for it in range(max_iter):
            changes = 0
            for c in self.classes:
                best_s = self.sigmas[c]
                best_score = self.evaluate_loo()
                
                # 在 [0.1, 3.0] 范围内进行线性搜索
                search_space = np.linspace(0.1, 3.0, 15)
                
                current_best_s = best_s
                
                for s in search_space:
                    self.sigmas[c] = s
                    score = self.evaluate_loo()
                    if score > best_score:
                        best_score = score
                        current_best_s = s
                
                # 如果 Sigma 发生显著变化，记录改变
                if abs(current_best_s - best_s) > 1e-3:
                    changes += 1
                
                self.sigmas[c] = current_best_s
            
            print(f"Iteration {it+1}: Sigmas = {self.sigmas}, Score = {best_score:.4f}")
            if changes == 0:
                break
                
    def evaluate_loo(self):
        """
        留一法交叉验证评估
            计算当前 Sigma 配置下的留一法准确率。
            
        返回值:
            float: 准确率 (0.0 - 1.0)
        """
        n = len(self.X_train)
        correct = 0
        
        # 预计算平方欧氏距离: (N, N)
        dists = cdist(self.X_train, self.X_train, 'sqeuclidean')
        
        for i in range(n):
            true_cls = self.y_train[i]
            scores = {}
            
            for c in self.classes:
                sigma = self.sigmas[c]
                # 获取属于类别 c 的样本索引
                c_indices = np.where(self.y_train == c)[0]
                # 排除自身 (LOO)
                c_indices = c_indices[c_indices != i]
                
                if len(c_indices) == 0:
                    scores[c] = 0
                    continue
                
                # 样本 i 到类别 c 所有训练样本的距离
                d = dists[i, c_indices]
                
                # 核函数: exp( -dist / 2sigma^2 )
                # 注意: 分母 (2*pi*sigma^2)^(d/2) 在比较不同类别时很重要，
                # 因为不同类别的 sigma 可能不同，所以必须包含归一化因子。
                # 特征维度 Dim = 12
                dim = self.X_train.shape[1]
                norm = (2 * np.pi * sigma**2) ** (dim / 2)
                
                kernels = np.exp(-d / (2 * sigma**2))
                # Parzen 窗密度估计: Sum(kernels) / (N * norm)
                density = np.sum(kernels) / (len(c_indices) * norm)
                scores[c] = density
            
            # 预测: 选择得分(密度)最高的类别
            if not scores:
                pred = self.classes[0]
            else:
                pred = max(scores, key=scores.get)
            
            if pred == true_cls:
                correct += 1
                
        return correct / n

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
