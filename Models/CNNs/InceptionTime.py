import torch.nn as nn
from tsai.models.InceptionTime import InceptionTime as InceptionTimeModel

from Models.CNNs.ResNet import CateEncoder


acts = {"relu": nn.modules.activation.ReLU}


class InceptionTime(nn.Module):
    def __init__(
        self,
        c_in,
        c_out,
        seq_len,
        nf,
        nb_filters,
        ks,
        bottleneck
    ) -> None:
        super().__init__()
        self.cate_encoder = CateEncoder(c_in)
        self.nume_encoder = nn.Conv1d(10, c_in, 1, bias=False)
        self.demo_encoder = nn.Linear(4, c_in, False)
        self.InceptionTimePlus_model = InceptionTimeModel(
            c_in=c_in,
            c_out=c_out,
            seq_len=seq_len,
            nf=nf,
            nb_filters=nb_filters,
            ks=ks,
            bottleneck=bottleneck
        )
        self._extra_repr = f"{c_in},{c_out},{seq_len},{nf},{nb_filters},"
        self._extra_repr += f"{ks},{bottleneck},"
        self.reset_parameters()

    def reset_parameters(self):
        print("reset parameters")
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, cate, nume, demo):
        x_cate = self.cate_encoder(cate)
        x_nume = self.nume_encoder(nume)
        x_demo = self.demo_encoder(demo).unsqueeze(2)

        h = x_cate + x_nume + x_demo
        return self.InceptionTimePlus_model(h)

    def extra_repr(self):
        return self._extra_repr
