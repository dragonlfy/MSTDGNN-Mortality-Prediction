import torch
from torch import nn
from torch.nn.init import xavier_uniform_

from Experiment.hcv_reader import HCV_Num_Possible_Values


def conv5x1(in_channels, out_channels, stride=1):
    """5x1 convolution with padding"""
    return nn.Conv1d(in_channels, out_channels, kernel_size=5,
                     stride=stride, padding=2)


def conv1x1(in_channels, out_channels, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_channels, out_channels, kernel_size=1,
                     stride=stride, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, channels, dropout=0.):
        super(BasicBlock, self).__init__()
        self.channels = channels
        self.conv1 = conv5x1(channels, channels)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.conv2 = conv5x1(channels, channels)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        identity = x
        x_conv1 = self.conv1(x)
        x_bn1 = self.bn1(x_conv1)
        x_relu = self.relu(x_bn1)
        x_dropout = self.dropout(x_relu)
        x_conv2 = self.conv2(x_dropout)
        x_bn2 = self.bn2(x_conv2)
        x_plus = x_bn2 + identity
        out = self.relu(x_plus)
        return out


class CateEncoder(nn.Module):
    def __init__(self, out_dim) -> None:
        super().__init__()
        self.nns = nn.ModuleList()
        for in_dim in HCV_Num_Possible_Values.values():
            emb = nn.Embedding(in_dim, out_dim)
            self.nns.append(emb)

    def reset_parameters(self):
        for emb in self.nns:
            xavier_uniform_(emb.weight.data)

    def forward(self, labels):
        embed_list = []
        for idx, net in enumerate(self.nns):
            embed_list.append(net(labels[:, idx, :]))

        embed_stack = torch.stack(embed_list, 1)
        embed = embed_stack.mean(1).permute((0, 2, 1))
        return embed


class ResNet(nn.Module):

    def __init__(self, ch_list, numclass, dropout):
        self.ch_list = ch_list
        super(ResNet, self).__init__()
        self.cate_encoder = CateEncoder(ch_list[0])
        self.nume_encoder = nn.Conv1d(10, ch_list[0], 1, bias=False)
        self.demo_encoder = nn.Linear(4, ch_list[0], False)

        self.module_list = nn.ModuleList()
        for idx in range(len(ch_list)-1):
            block = nn.Sequential(
                BasicBlock(ch_list[idx], dropout),
                nn.Conv1d(ch_list[idx], ch_list[idx+1], 3, 2),
                nn.BatchNorm1d(ch_list[idx+1]),
                nn.ReLU(True),
                nn.Dropout(dropout)
            )
            self.module_list.append(block)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(ch_list[-1], numclass)
        self._extra_repr = f"{ch_list},{numclass},{dropout}"
        self.reset_parameters()

    def reset_parameters(self):
        print('reset parameters')
        self.cate_encoder.reset_parameters()
        self.nume_encoder.reset_parameters()
        self.demo_encoder.reset_parameters()
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, cate, nume, demo):
        x_cate = self.cate_encoder(cate)
        x_nume = self.nume_encoder(nume)
        x_demo = self.demo_encoder(demo).unsqueeze(2)

        h = x_cate + x_nume + x_demo

        for module in self.module_list:
            h = module(h)

        h2 = self.avgpool(h)
        h3 = self.dropout(h2).squeeze(2)
        h4 = self.fc(h3)
        return h4

    def extra_repr(self):
        return self._extra_repr
