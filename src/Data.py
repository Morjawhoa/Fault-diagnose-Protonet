import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class SequenceDataset(Dataset):
    def __init__(self, path, length, device=torch.device('cpu')):
        """
        Args:
            path (string): Path to the csv file.
            length (int): Length of the sequence to be returned.
        """
        # 读取CSV文件
        data = pd.read_csv(path)
        data = data.values

        self.labels = data[:, -1].astype(int)
        self.features = data[:, :-1]

        self.length = length
        self.feature_size = int(self.features.shape[-1])
        self.device = device

    def __len__(self):
        # 数据集的长度是总数据长度减去序列长度，再加1
        return len(self.features) - self.length + 1

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 确定起始点和结束点
        start_idx = idx
        end_idx = start_idx + self.length

        feature = self.features[start_idx:end_idx]
        label = self.labels[start_idx:end_idx]  # 使用序列的最后一个点作为标签

        # 将数据转换为Tensor，并将其移动到GPU（如果可用）
        feature = torch.tensor(feature, dtype=torch.float, device=self.device)
        label = torch.tensor(label, dtype=torch.int, device=self.device)

        return feature, label


if __name__ == '__main__':
    # 示例使用
    path = '../data/train.csv'
    length = 10  # 假设每次读取的序列长度为10
    dataset = SequenceDataset(path, length)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 遍历整个数据集
    for features, labels in dataloader:
        # 这里可以处理每个batch的数据
        print(features.shape)
        print(labels.shape)
        break
