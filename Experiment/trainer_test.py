from typing import Dict, Mapping
from datetime import datetime
import time
import os.path as osp
import torch
from torchtnt.utils.loggers import CSVLogger, TensorBoardLogger
from torchtnt.utils.loggers.logger import Scalar
from tqdm import tqdm
import torch.nn.functional as F
from Experiment.config import ExperimentConfig
from Experiment.datautils import DLoader


def extract_edge_weights(model):
    # 假设 model 是你已经训练好的 GNN 模型实例
    # 提取 multi_shallow_embedding 模块
    embedding_module = model.g_constr

    # 使用 embedding_module 的参数生成邻接矩阵
    # 注意: 这里直接使用了 embedding_module.forward() 方法的实现逻辑
    emb_s = embedding_module.emb_s  # 形状: [num_graphs, num_nodes, 1]
    emb_t = embedding_module.emb_t  # 形状: [num_graphs, 1, num_nodes]

    # 计算邻接矩阵
    # 形状: [num_graphs, num_nodes, num_nodes]
    adj_matrix = torch.matmul(emb_s, emb_t)

    return adj_matrix


def extract_and_sort_edge_weights(model):
    # 提取边权重
    adj_matrix = extract_edge_weights(model)  # 使用前面定义的 extract_edge_weights 函数

    # 使用 softmax 归一化边权重
    normalized_weights = F.softmax(adj_matrix, dim=-1)

    num_graphs, num_nodes, _ = normalized_weights.shape

    sorted_weights_all_graphs = []
    sorted_edges_all_graphs = []

    for g in range(num_graphs):
        # 提取单个图的邻接矩阵
        adj = adj_matrix[g]

        # 展平邻接矩阵并排序
        flattened_adj = adj.view(-1)
        sorted_weights, sorted_indices = torch.sort(
            flattened_adj, descending=True)

        # 计算排序后的边的原始索引（即节点对）
        sorted_edges = [(idx.item() // num_nodes, idx.item() %
                         num_nodes) for idx in sorted_indices]

        sorted_weights_all_graphs.append(sorted_weights)
        sorted_edges_all_graphs.append(sorted_edges)

    return sorted_weights_all_graphs, sorted_edges_all_graphs


class Trainer:
    def __init__(
        self, window, cfg: ExperimentConfig
    ) -> None:
        self.window = window
        self.cfg = cfg
        self.device = cfg.device
        self.optimizer = cfg.optimizer
        self.loss_fn = cfg.lossfn
        self.num_epoches = cfg.num_epoches
        self.metrics: Dict = cfg.metrics
        self.relabel_w = cfg.relabel_w
        self.main_metric = cfg.main_metric
        self.results: Dict = {
            f'BestValid/{self.main_metric}': -1., f'Test/{self.main_metric}': -1.}
        self.splits = ["Train", "Valid", "Test"]
        while True:
            current_date = datetime.now()
            string_date = current_date.strftime(r"%m%d%H%M%S")
            self.log_path = f"logs/{cfg.dataset}/{cfg.logkey}/{string_date}_{window}"
            if osp.exists(self.log_path):
                time.sleep(2)
            else:
                break
        self.csv_logger = CSVLogger(self.log_path+'.csv', 1, True)
        self.board_logger = TensorBoardLogger(self.log_path)
        self.epoch_idx = 0
        self.save_cfg()

    @torch.no_grad()
    def infer_and_relabel(self, model, nextwind_loader: DLoader):
        model.load_state_dict(torch.load(self.log_path+'/best.pt'))
        model.eval()
        tqdm_bar = tqdm(
            enumerate(nextwind_loader),
            total=len(nextwind_loader),
            desc=f"{self.window}# Inferring  - ",
            dynamic_ncols=True,
            leave=True,
        )
        P_list = []
        for idx, batch in tqdm_bar:
            batch = [v.to(self.device) for v in batch]
            P = model.forward(*batch[:3]).detach().cpu()
            P_list.append(P)

        logits = torch.cat(P_list, 0)
        Ps = logits.sigmoid()
        nextwind_loader.relalel(Ps, self.relabel_w)

    def train_loop(self, model, train_loader):
        model.train()
        tqdm_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"{self.window}# Training         - {self.epoch_idx:03d}/{self.num_epoches:03d}",
            dynamic_ncols=True,
            leave=False,
        )

        Y_list = []
        P_list = []
        for idx, batch in tqdm_bar:
            batch = [v.to(self.device) for v in batch]
            P = model.forward(*batch[:3])
            Y = batch[-1]
            loss: torch.Tensor = self.loss_fn(P, Y.float())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            loss_val = loss.item()
            tqdm_bar.set_postfix(loss=f"{loss_val:.3f}")
            Y_list.append(Y)
            P_list.append(P.detach())

        Y_all = torch.cat(Y_list, 0)
        P_all = torch.cat(P_list, 0)
        avg_loss = self.loss_fn(P_all, Y_all.float()).item()

        self.last_results = self.results
        self.results = {"Training/Loss": avg_loss}

    @torch.no_grad()
    def eval_loop(self, split, model, dataloader):
        model.eval()
        tqdm_bar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"{self.window}# Evaluating {split:5s} - {self.epoch_idx:03d}/{self.num_epoches:03d}",  # noqa
            dynamic_ncols=True,
            leave=False,
        )
        Y_list = []
        P_list = []
        for idx, batch in tqdm_bar:
            batch = [v.to(self.device) for v in batch]
            P = model.forward(*batch[:3])
            Y = batch[-1]
            loss: torch.Tensor = self.loss_fn(P, Y.float())
            loss_val = loss.item()
            tqdm_bar.set_postfix(loss=f"{loss_val:.3f}")
            Y_list.append(Y)
            P_list.append(P.detach())

        # 假设 adj_matrix 是你的模型中的邻接矩阵，形状为 [4, 19, 19]
        sorted_weights_all_graphs, sorted_edges_all_graphs = extract_and_sort_edge_weights(
            model)
        # 打印每个图的前N个最大边权重及其对应的边
        N = 10
        # for g in range(len(sorted_weights_all_graphs)):
        #     print(f"Graph {g}:")
        #     for i in range(N):
        #         weight = sorted_weights_all_graphs[g][i].item()
        #         edge = sorted_edges_all_graphs[g][i]
        #         print(f"  Edge {edge} has weight {weight:.4f}")
        #     print("\n")
        with open('extract_and_sort_edge_weights_8.txt', 'a') as file:
            for g in range(len(sorted_weights_all_graphs)):
                file.write(f"Graph {g}:\n")
                for i in range(N):
                    if i < len(sorted_weights_all_graphs[g]):  # 确保不会因为N过大而出现索引越界
                        weight = sorted_weights_all_graphs[g][i]
                        edge = sorted_edges_all_graphs[g][i]
                        file.write(f"  Edge {edge} has weight {weight:.4f}\n")
                file.write("\n")

        Y_all = torch.cat(Y_list, 0)
        P_all = torch.cat(P_list, 0)
        avg_loss = self.loss_fn(P_all, Y_all.float()).item()
        P_all, Y_all = P_all.detach().cpu(), Y_all.cpu() > 0.5
        self.results[f"{split}/Loss"] = avg_loss
        for key, metric_fn in self.metrics.items():
            P_all[torch.where(P_all.isnan())] = 0
            self.results[f"{split}/{key}"] = metric_fn(Y_all, P_all)
        pass

    def train(self, model: torch.nn.Module,
              train_loader: DLoader, valid_loader: DLoader,
              test_loader: DLoader):

        for epoch_idx in range(1, self.num_epoches + 1):
            self.epoch_idx = epoch_idx
            self.train_loop(model, train_loader)
            self.eval_loop("Train", model, train_loader)
            self.eval_loop("Valid", model, valid_loader)
            if self.improved():
                torch.save(model.state_dict(), self.log_path+'/best.pt')
                if test_loader is not None:
                    self.eval_loop("Test", model, test_loader)
            else:
                self.repeat_best_results()

            self.log_dict(self.results)
            res_tuple = [f"{key:5s}: {val*100.:0>5.2f}"
                         for key, val in self.results.items()
                         if (self.main_metric in key)]
            res_str = "   ".join(res_tuple)
            print(f"{self.window}# {self.epoch_idx:03d}/{self.num_epoches:03d} |", res_str)  # noqa

        self.log_close()

    def eval_early_addmission(
            self, model, train_loader, valid_loader, test_loader):
        self.eval_loop("Train", model, train_loader)
        self.eval_loop("Valid", model, valid_loader)
        self.eval_loop("Test", model, test_loader)
        self.log_close()

    def log_dict(self, payload: Mapping[str, Scalar]):
        self.csv_logger.log_dict(payload, self.epoch_idx)
        self.board_logger.log_dict(payload, self.epoch_idx)

    def log_close(self):
        self.csv_logger.close()
        self.board_logger.close()

    def repeat_best_results(self):
        for key, val in self.last_results.items():
            if key.startswith(('Test', 'Best')):
                self.results[key] = val

    def improved(self):
        if self.results[f"Valid/{self.main_metric}"] > self.last_results[f"BestValid/{self.main_metric}"]:
            for key in self.metrics.keys():
                self.results[f"BestValid/{key}"] = self.results[f"Valid/{key}"]
            self.results["BestEpoch"] = self.epoch_idx
            return True
        else:
            self.results[f"BestValid/{self.main_metric}"] = \
                self.last_results[f"BestValid/{self.main_metric}"]
            return False

    def save_cfg(self):
        with open(self.log_path+'/cfg.txt', 'w') as wfile:
            for key, val in self.cfg.__dict__.items():
                print(key, val, file=wfile)
