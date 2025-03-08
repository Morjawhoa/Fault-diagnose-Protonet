import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_


def time_divide(datas, labels, length):
    max_start_index = datas.shape[0] - length
    # 随机选择一个起点
    start_index = torch.randint(0, max_start_index + 1, (1,)).item()
    # 截取子序列
    datas = datas[start_index:start_index + length]
    labels = labels[start_index:start_index + length]
    return datas, labels


def list_to_dict(datas, labels):  # 将一堆尚未分类的数据，往往是已经经过lstm输出过得到的原型向量，以字典形式进行分类
    data_labels = {}
    for data, label in zip(datas, labels):
        if label not in data_labels.keys():
            data_labels[label] = data.unsqueeze(0)
        else:
            data_labels[label] = torch.cat((data_labels[label], data.unsqueeze(0)), dim=0)
    return data_labels


def random_sample(data_set):  # 从D_set随机对半取支持集和查询集
    # 生成一个随机排列的索引
    half = int(data_set.shape[0] / 2)
    shuffled_indices = torch.randperm(data_set.shape[0])
    # 使用这些索引来索引张量
    data_set = data_set[shuffled_indices]
    query = data_set[half:]
    support = data_set[:half+1]
    return query, support


def my_protonet(input_size, hidden_size, num_layers, output_size, path_model, device=torch.device('cpu'), step=None):
    if step is None:
        model = MyProtonet(input_size, hidden_size, num_layers, output_size, device=device)
    else:
        model = torch.load(path_model + f'step_{step}.pkl')
    return model


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, device=torch.device('cpu')):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=0.5)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.5)
        self.device = device
        self.to(device)

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, 1, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, 1, self.hidden_size).to(self.device)
        # 前向传播LSTM
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        out = self.fc(out)
        return out


class MyProtonet:
    def __init__(self, input_size, hidden_size, num_layers, output_size, device=torch.device('cpu')):
        self.input_size = input_size
        self.output_size = output_size
        self.batchSize = 1

        self.rnn = LSTMModel(input_size, hidden_size, num_layers, output_size, device=device)

        self.class_to_idx = {}
        self.centers_dict = {}

        self.optimizer = torch.optim.Adam(self.rnn.parameters(), lr=0.001)
        self.device = device

    def get_center_query(self, outputs, labels):  # 输入字典，返回一个划分有中心、支持集、标签的高级字典
        data_labels = list_to_dict(outputs, labels)
        queries, labels, centers = [], [], []
        for i, key in enumerate(data_labels.keys()):
            # 从data_set随机取支持集和查询集
            query, support = random_sample(data_labels[key])
            label = torch.tensor([i] * query.shape[0]).to(self.device)
            center = support.mean(dim=0)
            # 更新原型向量
            self.centers_dict[key] = center
            self.class_to_idx[key] = i
            # 将支持集和查询集存储在list中
            queries.append(query)
            labels.append(label)
            centers.append(center)
        queries = torch.cat(queries, dim=0)
        labels = torch.cat(labels, dim=0)
        centers = torch.stack(centers, dim=0)
        return queries, labels, centers

    def train(self, features, labels):  # 网络的训练
        self.rnn.train()
        # 前向传播，映射为原型向量
        outputs = self.rnn(features.unsqueeze(1)).squeeze(1)
        # 分类
        queries, labels, centers = self.get_center_query(outputs, labels)
        # loss
        dist = torch.cdist(queries, centers)
        dist = F.log_softmax(-dist, dim=-1)
        dist = torch.gather(dist, -1, labels.unsqueeze(-1))
        loss = -dist.mean()
        # 优化器
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.rnn.parameters(), 1)
        self.optimizer.step()
        return loss

    def predict(self, features):  # 网络的预测
        self.rnn.eval()
        centers = torch.stack(list(self.centers_dict.values()), dim=0)
        keys = list(self.centers_dict.keys())
        # 前向传播，映射为原型向量
        outputs = self.rnn(features.unsqueeze(1)).squeeze(1)
        dist = torch.cdist(outputs, centers)
        dist = F.log_softmax(-dist, dim=-1)
        argmax = torch.argmax(dist, dim=-1).tolist()
        predicts = torch.tensor([keys[i] for i in argmax], dtype=torch.int, device=self.device)
        return predicts
