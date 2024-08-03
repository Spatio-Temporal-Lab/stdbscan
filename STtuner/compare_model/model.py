import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class PairwiseComparisonModel:
    def __init__(self, input_dim=38, learning_rate=0.001):
        """
        初始化模型

        :param input_dim: 输入特征的维度
        :param learning_rate: 学习率
        """
        self.model = self._build_model(input_dim)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def _build_model(self, input_dim):
        """
        构建模型

        :param input_dim: 输入特征的维度
        :return: 神经网络模型
        """

        class Network(nn.Module):
            def __init__(self, input_dim):
                super(Network, self).__init__()
                self.shared_network = nn.Sequential(
                    nn.Linear(input_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 16),
                    nn.ReLU()
                )
                self.output_layer = nn.Linear(16, 1)
                self.sigmoid = nn.Sigmoid()

            def forward(self, x1, x2):
                output1 = self.shared_network(x1)
                output2 = self.shared_network(x2)
                difference = torch.abs(output1 - output2)  # 计算绝对差异
                output = self.output_layer(difference)
                return self.sigmoid(output)

        return Network(input_dim)

    def generate_pairwise_data(self, df, result_col='result'):
        """
        生成成对的数据

        :param df: 输入DataFrame
        :param result_col: 表示质量的列名
        :return: 成对的输入数据和标签
        """
        num_samples = len(df)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        pair_data1 = []
        pair_data2 = []
        pair_labels = []

        for i in range(0, num_samples - 1, 2):
            idx1, idx2 = indices[i], indices[i + 1]

            row1 = df.iloc[idx1]
            row2 = df.iloc[idx2]

            data1 = row1.drop(result_col).values
            data2 = row2.drop(result_col).values

            if row1[result_col] < row2[result_col]:
                label = 1
            else:
                label = 0

            pair_data1.append(data1)
            pair_data2.append(data2)
            pair_labels.append(label)

        return np.array(pair_data1), np.array(pair_data2), np.array(pair_labels)

    def fit(self, df, result_col='result', epochs=100, batch_size=32, test_size=0.2, random_state=22):
        """
        训练模型

        :param df: 输入DataFrame
        :param result_col: 表示质量的列名
        :param epochs: 训练轮数
        :param batch_size: 每个批次的大小
        :param test_size: 测试集比例
        :param random_state: 随机种子
        """
        pair_data1, pair_data2, pair_labels = self.generate_pairwise_data(df, result_col)

        # 转换为Tensor
        pair_data1 = torch.tensor(pair_data1, dtype=torch.float32)
        pair_data2 = torch.tensor(pair_data2, dtype=torch.float32)
        pair_labels = torch.tensor(pair_labels, dtype=torch.float32)

        # 分割数据
        data1_train, data1_test, data2_train, data2_test, labels_train, labels_test = train_test_split(
            pair_data1, pair_data2, pair_labels, test_size=test_size, random_state=random_state)

        # 训练模型
        for epoch in range(epochs):
            self.model.train()
            permutation = torch.randperm(data1_train.size()[0])

            for i in range(0, data1_train.size()[0], batch_size):
                indices = permutation[i:i + batch_size]
                batch_data1, batch_data2, batch_labels = data1_train[indices], data2_train[indices], labels_train[
                    indices]

                self.optimizer.zero_grad()
                outputs = self.model(batch_data1, batch_data2).squeeze()
                loss = self.criterion(outputs, batch_labels)
                loss.backward()
                self.optimizer.step()

            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

        self.data1_test = data1_test
        self.data2_test = data2_test
        self.labels_test = labels_test

    def evaluate(self):
        """
        评估模型
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.data1_test, self.data2_test).squeeze()
            predicted_labels = (outputs > 0.5).float()
            accuracy = accuracy_score(self.labels_test.numpy(), predicted_labels.numpy())
            print(f'Test Accuracy: {accuracy:.4f}')

    def predict(self, x1, x2):
        """
        对新数据进行预测

        :param x1: 第一个输入数据
        :param x2: 第二个输入数据
        :return: 预测结果
        """
        self.model.eval()
        with torch.no_grad():
            x1 = torch.tensor(x1, dtype=torch.float32)
            x2 = torch.tensor(x2, dtype=torch.float32)
            output = self.model(x1.unsqueeze(0), x2.unsqueeze(0)).squeeze()
            return output.item()


