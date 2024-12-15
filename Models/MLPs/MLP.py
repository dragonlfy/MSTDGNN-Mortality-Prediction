import torch.nn as nn
from tsai.models.MLP import MLP

from Models.CNNs.ResNet import CateEncoder


acts = {
    "relu": nn.ReLU(inplace=True)
}


class MLPModel(nn.Module):
    def __init__(
        self,
        c_emb,
        c_out,
        seq_len,
        layers,
        ps,
        act,
        use_bn,
        bn_final,
        lin_first,
        fc_dropout,
        y_range,
    ) -> None:
        super().__init__()
        self.cate_encoder = CateEncoder(c_emb)
        self.nume_encoder = nn.Conv1d(10, c_emb, 1, bias=False)
        self.demo_encoder = nn.Linear(4, c_emb, False)
        act = acts[act]
        self.MLP_model = MLP(
            c_in=c_emb,
            c_out=c_out,
            seq_len=seq_len,
            layers=layers,
            ps=ps,
            act=act,
            use_bn=use_bn,
            bn_final=bn_final,
            lin_first=lin_first,
            fc_dropout=fc_dropout,
            y_range=y_range,
        )
        self.reset_parameters()
        self._extra_repr = f"{c_emb},{c_out},{seq_len},{layers},{ps},{act},"
        self._extra_repr += f"{use_bn},{bn_final},{lin_first},{fc_dropout},{y_range}"

    def reset_parameters(self):
        print('reset parameters')
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
        return self.MLP_model(h)

    def extra_repr(self):
        return self._extra_repr
