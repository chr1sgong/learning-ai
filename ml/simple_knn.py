import numpy as np
from collections import Counter

class SimpleKNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, x, y):
        """存储训练数据"""
        self.x_train = x
        self.y_train = y
    
    def predict(self, X):
        """预测"""
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        # 计算距离
        distances = [np.linalg.norm(x - x_train) for x_train in self.x_train]

        # 获取最近的k个样本的索引
        k_indices = np.argsort(distances)[:self.k]

        # 获取k个最近邻的标签
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # 多数投票
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    
if __name__ == "__main__":
    print("test")