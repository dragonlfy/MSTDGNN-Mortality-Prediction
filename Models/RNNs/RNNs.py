import torch.nn as nn
from tsai.models.RNN import RNN

from Models.CNNs.ResNet import CateEncoder


class RNNModel(nn.Module):
    def __init__(
        self,
        c_emb,
        c_out,
        hidden_size,
        n_layers,
        bias,
        rnn_dropout,
        bidirectional,
        fc_dropout,
        init_weights,
    ) -> None:
        super().__init__()
        self.cate_encoder = CateEncoder(c_emb)
        self.nume_encoder = nn.Conv1d(10, c_emb, 1, bias=False)
        self.demo_encoder = nn.Linear(4, c_emb, False)
        self.RNN_model = RNN(
            c_in=c_emb,
            c_out=c_out,
            hidden_size=hidden_size,
            n_layers=n_layers,
            bias=bias,
            rnn_dropout=rnn_dropout,
            bidirectional=bidirectional,
            fc_dropout=fc_dropout,
            init_weights=init_weights,
        )
        self.reset_parameters()

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
        return self.RNN_model(h)


class RNNPlusModel(nn.Module):
    def __init__(
        self,
        c_emb,
        c_out,
        seq_len,
        hidden_size,
        n_layers,
        bias,
        rnn_dropout,
        bidirectional,
        n_cat_embeds,
        cat_embed_dims,
        cat_padding_idxs,
        cat_pos,
        feature_extractor,
        fc_dropout,
        last_step,
        bn,
        custom_head,
        y_range,
        init_weights,
    ) -> None:
        super().__init__()
        self.cate_encoder = CateEncoder(c_emb)
        self.nume_encoder = nn.Conv1d(10, c_emb, 1, bias=False)
        self.demo_encoder = nn.Linear(4, c_emb, False)
        self.RNNPlus_model = RNN(
            c_in=c_emb,
            c_out=c_out,
            seq_len=seq_len,
            hidden_size=hidden_size,
            n_layers=n_layers,
            bias=bias,
            rnn_dropout=rnn_dropout,
            bidirectional=bidirectional,
            fc_dropout=fc_dropout,
            init_weights=init_weights,
            n_cat_embeds=n_cat_embeds,
            cat_embed_dims=cat_embed_dims,
            cat_padding_idxs=cat_padding_idxs,
            cat_pos=cat_pos,
            feature_extractor=feature_extractor,
            last_step=last_step,
            bn=bn,
            custom_head=custom_head,
            y_range=y_range,
        )
        self.reset_parameters()
        self._extra_repr = f"{c_emb},{c_out},{seq_len},{hidden_size},{n_layers},{bias},{rnn_dropout},"
        self._extra_repr += f"{bidirectional},{n_cat_embeds},{cat_embed_dims},{cat_padding_idxs},{cat_pos},"
        self._extra_repr += f"{feature_extractor},{fc_dropout},{last_step},{bn},{custom_head},{y_range},{init_weights}"

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
        return self.RNNPlus_model(h)

    def extra_repr(self):
        return self._extra_repr
