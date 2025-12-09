import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding='same'):
        super(DepthwiseSeparableConv1d, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class WL(nn.Module):
    def __init__(self, num_classes, num_features,
                 conv1_nf, conv2_nf, fc_drop_p, lstm_hidden=4):
        super(WL, self).__init__()

        self.num_classes = num_classes
        self.num_features = num_features
        self.conv1_nf = conv1_nf
        self.conv2_nf = conv2_nf
        self.fc_drop_p = fc_drop_p

        # 卷积层
        self.conv1 = DepthwiseSeparableConv1d(num_features, conv1_nf, 1)
        self.conv2 = DepthwiseSeparableConv1d(conv1_nf, conv2_nf, 3)

        # 批归一化
        self.bn1 = nn.BatchNorm1d(conv1_nf)
        self.bn2 = nn.BatchNorm1d(conv2_nf)

        # BiLSTM层
        self.bilstm = nn.LSTM(
            input_size=conv2_nf,
            hidden_size=lstm_hidden,
            bidirectional=True,
            batch_first=True
        )

        # 全连接层
        self.fc = nn.Linear(2 * lstm_hidden, num_classes)  # 双向输出维度为2*hidden_size

        # 激活函数和Dropout
        self.relu = nn.ReLU()
        self.convDrop = nn.Dropout(fc_drop_p)

    def forward(self, x):
        # 输入x形状: (batch_size, seq_len, num_features)
        x = x.transpose(1, 2)  # 转换为(batch, features, seq_len)供Conv1d使用

        # 第一层卷积
        x = self.conv1(x)
        x = self.relu(self.bn1(x))
        x = self.convDrop(x)

        # 第二层卷积
        x = self.conv2(x)
        x = self.relu(self.bn2(x))
        x = self.convDrop(x)  # 输出形状: (batch, conv2_nf, seq_len)

        # 调整维度并输入BiLSTM
        x = x.transpose(1, 2)  # 转换为(batch, seq_len, conv2_nf)
        x, _ = self.bilstm(x)  # 输出形状: (batch, seq_len, 2*lstm_hidden)

        # 全局平均池化
        x = torch.mean(x, dim=1)  # 输出形状: (batch, 2*lstm_hidden)

        # 全连接层
        out_log = self.fc(x)
        output = F.softmax(out_log, dim=1)

        return out_log, output