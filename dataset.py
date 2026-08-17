import numpy as np
import h5py
import torch
from torch.utils.data import Dataset

class ResistivityPreprocessor2D:
    """2D电阻率数据预处理类：支持对数变换+Z-Score归一化"""
    def __init__(self, use_log=True, epsilon=1e-8):
        self.use_log = use_log
        self.epsilon = epsilon
        self.mean = None
        self.std = None

    def _log_transform(self, data):
        if (data < 0).any():
            raise ValueError("原始数据包含负值，无法进行对数变换！")
        return np.log10(data + self.epsilon)

    def _inverse_log_transform(self, log_data):
        return 10 ** log_data - self.epsilon

    def fit(self, train_data):
        """train_data形状：(样本数, 通道数, 高度, 宽度)"""
        transformed_data = self._log_transform(train_data) if self.use_log else train_data.copy()
        self.mean = np.mean(transformed_data, axis=(0, 2, 3))  # 保留通道维度
        self.std = np.std(transformed_data, axis=(0, 2, 3)) + self.epsilon

    def transform(self, data):
        if self.mean is None or self.std is None:
            raise RuntimeError("请先调用fit()拟合训练集！")
        transformed_data = self._log_transform(data) if self.use_log else data.copy()
        for c in range(data.shape[1]):
            transformed_data[:, c] = (transformed_data[:, c] - self.mean[c]) / self.std[c]
        return transformed_data

    def inverse_transform(self, normalized_data):
        data = normalized_data.copy()
        for c in range(data.shape[1]):
            data[:, c] = data[:, c] * self.std[c] + self.mean[c]
        return self._inverse_log_transform(data) if self.use_log else data


class ResistivityDataset2D(Dataset):
    def __init__(self, mat_path, sample_key, label_key, preprocessor=None, split_width=5):
        self.preprocessor = preprocessor
        self.split_width = split_width

        # 读取MAT数据
        with h5py.File(mat_path, 'r') as f:
            raw_samples = f[sample_key][()]
            raw_labels = f[label_key][()]
        #print(f"原始样本维度: {raw_samples.shape}，原始标签维度: {raw_labels.shape}")

        # 调整轴顺序为(样本数, 高度, 宽度)
        samples = np.transpose(raw_samples, (0, 2, 1)).astype(np.float32)
        labels = np.transpose(raw_labels, (0, 2, 1)).astype(np.float32)
        #print(f"转换后samples形状: {samples.shape}（样本数, 高度, 宽度）")

        # 分割处理
        self.split_samples, self.split_labels = [], []
        for sample, label in zip(samples, labels):
            height, width = sample.shape
            num_splits = width // self.split_width  # 整数除法丢弃余数
            for i in range(num_splits):
                start, end = i * self.split_width, (i + 1) * self.split_width
                # 添加通道维度 (高度, 宽度) → (1, 高度, 宽度)
                self.split_samples.append(sample[:, start:end][np.newaxis, ...])
                self.split_labels.append(label[:, start:end][np.newaxis, ...])

        # 转换为numpy数组
        self.split_samples = np.array(self.split_samples, dtype=np.float32)
        self.split_labels = np.array(self.split_labels, dtype=np.float32)

        # 预处理
        if preprocessor is not None:
            self.processed_samples = preprocessor.transform(self.split_samples)
            self.processed_labels = preprocessor.transform(self.split_labels)
        else:
            self.processed_samples = self.split_samples
            self.processed_labels = self.split_labels

    def __len__(self):
        return len(self.processed_samples)

    def __getitem__(self, idx):
        return torch.tensor(self.processed_samples[idx]), torch.tensor(self.processed_labels[idx])