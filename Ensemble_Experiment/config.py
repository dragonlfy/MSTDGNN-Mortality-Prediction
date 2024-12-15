from typing import Dict

import numpy as np
import torch
from sklearn import metrics

from Models.CNNs.ResNet import ResNet
from Models.Transformers.Transformer import Transformer
from Models.CNNs.ResNetPlus import ResNetPlus
from Models.CNNs.InceptionTimePlus import InceptionTimePlus
from Models.RNNs.LSTMAttention import LSTMAttention
from Models.MLPs.MLP import MLPModel
from Models.MLPs.gMLP import GMLP
from Models.GNN.GNNStack import GNNStack
from Models.RNNs.RNNs import RNNModel


model_zoo = {
    "resnet": ResNet,
    "transformer": Transformer,
    "resnetplus": ResNetPlus,
    "inceptiontimeplus": InceptionTimePlus,
    "lstmattention": LSTMAttention,
    "mlp": MLPModel,
    "gmlp": GMLP,
    "rnn": RNNModel,
    "gnnstack": GNNStack,
}


class ExperimentConfig():
    @property
    def datakey(self) -> str:
        repr = f"DPGap{self.death_pred_gap}"
        repr += f"SPGap{self.survival_pred_gap}"
        repr += f"Len{self.segment_length}"
        repr += f"Str{self.segment_stride}"
        repr += f"Art{self.artifact_length}"
        repr += f"Grp{self.num_groups}"
        return repr

    @property
    def logkey(self):
        return self._logkey

    @property
    def datapath(self):
        return f'{self.segment_folder}/{self.datakey}_hf.pkl'

    @property
    def datamodule(self):
        return self._datam

    @property
    def model(self):
        return self._model

    @property
    def models(self):
        return self._models

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def optimizers(self):
        return self._optimizers

    @property
    def lossfn(self):
        return self._lossfn

    @property
    def lossfns(self):
        return self._lossfns

    @property
    def metrics(self) -> Dict:
        return self._metrics

    def __init__(self, dataset, data=None, model=None, model_args=None, n_fold=None, i_fold=None):
        assert dataset in {'mimic3', 'mimic4'}
        self.dataset = dataset
        self.n_fold = n_fold
        self.i_fold = i_fold
        self.batch_size = 64
        self.num_epoches = 100
        self.num_workers = 0
        self._metrics = {
            'ACC': lambda y, p: metrics.accuracy_score(y, p > 0),
            'AUROC': lambda y, p:
                metrics.roc_auc_score(y, p.sigmoid()),
            'F1': lambda y, p:
                metrics.f1_score(y, p > 0, zero_division=0.),
            'Fbeta': lambda y, p:
                metrics.fbeta_score(y, p > 0, beta=np.sqrt(11.7), zero_division=0.),
            'Precision': lambda y, p:
                metrics.precision_score(y, p > 0, zero_division=0.),
            'Recall': lambda y, p:
                metrics.recall_score(y, p > 0, zero_division=0.),
            'AUPRC': lambda y, p:
                metrics.average_precision_score(y, p.sigmoid()),
        }
        self.main_metric = "Fbeta"
        # 通过传参设置 config
        if data is None:
            self._____DPGap4SPGap48Len24Str12Art2Grp8()
        elif data == 'DPGap4SPGap48Len24Str12Art2Grp8_hf':
            self._____DPGap4SPGap48Len24Str12Art2Grp8()

        if model is not None:
            if self.dataset == 'mimic3':
                from Experiment.mimic3 import segment_folder
                from Experiment.datautils import MIMIC3DataModule as DataModule
            else:
                from Experiment.mimic4 import segment_folder
                from Experiment.datautils4 import MIMIC4DataModule as DataModule

            self.segment_folder = segment_folder
            self._datam = DataModule(self)
            self._model_name = model
            self._model_args = model_args
            self.build_model()
            self._logkey = f"{self.datakey}/{model}_ensemble"

    def _____DPGap4SPGap48Len24Str12Art2Grp8(self) -> None:
        self.death_pred_gap = 4
        self.survival_pred_gap = 48
        self.segment_length = 24
        self.segment_stride = 12
        self.artifact_length = 2
        self.num_groups = 8
        self.relabel_w = 0.25

    def build_model(self) -> None:
        self._models = []
        self._lossfns = []
        self._optimizers = []
        for model_args in self._model_args:
            model = self._model_name
            self.device = torch.device('cuda:0')
            self._model = model_zoo[model.lower()](**model_args['net']).to(self.device)
            self._optimizer = torch.optim.AdamW(
                self._model.parameters(), **model_args['opt'])
            loss_args = model_args['loss'].copy()
            if 'pos_weight' in loss_args:
                loss_args['pos_weight'] = torch.tensor(loss_args['pos_weight']).to(self.device)
            self._lossfn = torch.nn.BCEWithLogitsLoss(**loss_args)
            self._models.append(self._model)
            self._lossfns.append(self._lossfn)
            self._optimizers.append(self._optimizer)
