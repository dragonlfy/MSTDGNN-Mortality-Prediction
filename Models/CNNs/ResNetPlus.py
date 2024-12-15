import torch.nn as nn
from tsai.models.ResNetPlus import ResNetPlus as ResNetPlusModel

from Models.CNNs.ResNet import CateEncoder

acts = {"relu": nn.modules.activation.ReLU}


class ResNetPlus(nn.Module):
    def __init__(
        self,
        c_in,
        c_out,
        seq_len,
        nf,
        sa,
        se,
        fc_dropout,
        concat_pool,
        flatten,
        custom_head,
        y_range,
        ks,
        coord,
        separable,
        bn_1st,
        zero_norm,
        act,
    ) -> None:
        super().__init__()
        self.cate_encoder = CateEncoder(c_in)
        self.nume_encoder = nn.Conv1d(10, c_in, 1, bias=False)
        self.demo_encoder = nn.Linear(4, c_in, False)
        self.resNetPlus_model = ResNetPlusModel(
            c_in=c_in,
            c_out=c_out,
            seq_len=seq_len,
            nf=nf,
            sa=sa,
            se=se,
            fc_dropout=fc_dropout,
            concat_pool=concat_pool,
            flatten=flatten,
            custom_head=custom_head,
            y_range=y_range,
            ks=ks,
            coord=coord,
            separable=separable,
            bn_1st=bn_1st,
            zero_norm=zero_norm,
            act=acts[act],
        )
        self._extra_repr = f"{c_in},{c_out},{seq_len},{nf},{sa},{se},{fc_dropout},"
        self._extra_repr += f"{concat_pool},{flatten},{custom_head},{y_range},{ks},"
        self._extra_repr += f"{coord},{separable},{bn_1st},{zero_norm},{act}"
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
        return self.resNetPlus_model(h)

    def extra_repr(self):
        return self._extra_repr
