import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

from MyProtonet import my_protonet
from Data import SequenceDataset

step = None  # 读档的步数，None就说明不读档
length = 500  # 每次切分出来的时间序列长度.
check_point = 50  # 每个进度条表示的epoch数
epochs = 10000  # 迭代次数

device = torch.device("cuda")
path_model = '../models/'
path_train = '../data/train.csv'
path_test = '../data/test.csv'
bar_format = "\033[94m{l_bar}\033[95m{bar}\033[94m{r_bar}\033[0m"

local_time = time.localtime(time.time())
formatted_time = time.strftime("%Y-%m-%d_%H-%M-%S", local_time)

# 载入数据
dataset_train = SequenceDataset(path_train, length, device=device)
dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=True)
dataset_test = SequenceDataset(path_test, length, device=device)
dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

input_size = dataset_train.feature_size
hidden_size = 10  # 隐藏层尺寸
num_layers = 2  # LSTM层数
output_size = 3  # 原型向量尺寸

# 初始化模型
model = my_protonet(input_size, hidden_size, num_layers, output_size, path_model, device=device, step=step)

accuracy_train = None
accuracy_test = None
pbar = tqdm(total=check_point, desc='0', bar_format=bar_format)
for n in range(epochs):
    features_train, labels_train = next(iter(dataloader_train))
    features_train, labels_train = features_train[0], labels_train[0]
    # 训练模型
    loss = model.train(features_train, labels_train)
    # 检查点
    if n % check_point == 0:
        # 保存模型
        torch.save(model, path_model + f'step_{n}.pkl')
        # 评估模型
        features_test, labels_test = next(iter(dataloader_test))
        features_test, labels_test = features_test[0], labels_test[0]
        predicts_train = model.predict(features_train)
        predicts_test = model.predict(features_test)
        accuracy_train = torch.eq(predicts_train, labels_train).float().mean().item()
        accuracy_test = torch.eq(predicts_test, labels_test).float().mean().item()
        # 记录评估
        str_data = f'{n},{accuracy_train},{accuracy_test}\n'
        with open(f'../logs/{formatted_time}.csv', "a") as f:
            f.write(str_data)
        # 下一个进度条
        if n != 0:
            pbar.set_postfix({'loss': loss.item(),
                              'accuracy_train': f"{accuracy_train:.2%}",
                              'accuracy_test': f"{accuracy_test:.2%}",
                              })
            del pbar
            pbar = tqdm(total=check_point, desc=str(n), bar_format=bar_format)
    # 更新进度条
    pbar.update(1)
    pbar.set_postfix({'loss': loss.item(),
                      })


