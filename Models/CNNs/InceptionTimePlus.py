import torch.nn as nn
from tsai.models.InceptionTimePlus import InceptionTimePlus as InceptionTimePlusModel

from Models.CNNs.ResNet import CateEncoder


acts = {"relu": nn.modules.activation.ReLU}


class InceptionTimePlus(nn.Module):
    def __init__(
        self,
        c_in,
        c_out,
        seq_len,
        nf,
        nb_filters,
        flatten,
        concat_pool,
        fc_dropout,
        bn,
        y_range,
        custom_head,
        ks,
        bottleneck,
        padding,
        coord,
        separable,
        dilation,
        stride,
        conv_dropout,
        sa,
        se,
        norm,
        zero_norm,
        bn_1st,
        act,
    ) -> None:
        super().__init__()
        self.cate_encoder = CateEncoder(c_in)
        self.nume_encoder = nn.Conv1d(10, c_in, 1, bias=False)
        self.demo_encoder = nn.Linear(4, c_in, False)
        self.InceptionTimePlus_model = InceptionTimePlusModel(
            c_in=c_in,
            c_out=c_out,
            seq_len=seq_len,
            nf=nf,
            nb_filters=nb_filters,
            flatten=flatten,
            concat_pool=concat_pool,
            fc_dropout=fc_dropout,
            bn=bn,
            y_range=y_range,
            custom_head=custom_head,
            ks=ks,
            bottleneck=bottleneck,
            padding=padding,
            coord=coord,
            separable=separable,
            dilation=dilation,
            stride=stride,
            conv_dropout=conv_dropout,
            sa=sa,
            se=se,
            norm=norm,
            zero_norm=zero_norm,
            bn_1st=bn_1st,
            act=acts[act],
        )
        self._extra_repr = f"{c_in},{c_out},{seq_len},{nf},{nb_filters},{flatten},{concat_pool},{fc_dropout},"
        self._extra_repr += f"{bn},{y_range},{custom_head},{ks},{bottleneck},{padding},{coord},{separable},{dilation},{stride},"
        self._extra_repr += f"{conv_dropout},{sa},{se},{norm},{zero_norm},{bn_1st},{act}"
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
