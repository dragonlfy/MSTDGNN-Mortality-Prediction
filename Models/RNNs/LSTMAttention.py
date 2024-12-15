import torch.nn as nn

from Models.CNNs.ResNet import CateEncoder


from tsai.models.RNNAttention import LSTMAttention as LSTMAttentionModel


class LSTMAttention(nn.Module):
    def __init__(
        self,
        c_in,
        c_out,
        seq_len,
        hidden_size,
        rnn_layers,
        bias,
        rnn_dropout,
        bidirectional,
        encoder_layers,
        n_heads,
        d_k,
        d_v,
        d_ff,
        encoder_dropout,
        act,
        fc_dropout,
        y_range,
        verbose,
        custom_head,
    ) -> None:
        super().__init__()
        self.cate_encoder = CateEncoder(c_in)
        self.nume_encoder = nn.Conv1d(10, c_in, 1, bias=False)
        self.demo_encoder = nn.Linear(4, c_in, False)
        self.LSTMAttention_model = LSTMAttentionModel(
            c_in=c_in,
            c_out=c_out,
            seq_len=seq_len,
            hidden_size=hidden_size,
            rnn_layers=rnn_layers,
            bias=bias,
            rnn_dropout=rnn_dropout,
            bidirectional=bidirectional,
            encoder_layers=encoder_layers,
            n_heads=n_heads,
            d_k=d_k,
            d_v=d_v,
            d_ff=d_ff,
            encoder_dropout=encoder_dropout,
            act=act,
            fc_dropout=fc_dropout,
            y_range=y_range,
            verbose=verbose,
            custom_head=custom_head,
        )
        self.reset_parameters()
        self._extra_repr = f"{c_in},{c_out},{seq_len},{hidden_size},{rnn_layers},{bias},{rnn_dropout},{bidirectional}"
        self._extra_repr += f",{encoder_layers},{n_heads},{d_k},{d_v},{d_ff},{encoder_dropout},{act}"
        self._extra_repr += f",{fc_dropout},{y_range},{verbose},{custom_head}"

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
        return self.LSTMAttention_model(h)

    def extra_repr(self):
        return self._extra_repr
