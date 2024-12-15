from typing import Dict, Mapping
from datetime import datetime
import time
import os.path as osp
import torch
from torchtnt.utils.loggers import CSVLogger, TensorBoardLogger
from torchtnt.utils.loggers.logger import Scalar
from tqdm import tqdm
from typing import List
from Ensemble_Experiment.config import ExperimentConfig
from Experiment.datautils import DLoader


class Trainer:
    def __init__(
        self, window, cfg: ExperimentConfig
    ) -> None:
        self.window = window
        self.cfg = cfg
        self.device = cfg.device
        self.optimizers = cfg.optimizers
        self.loss_fns = cfg.lossfns
        self.num_epoches = cfg.num_epoches
        self.metrics: Dict = cfg.metrics
        self.relabel_w = cfg.relabel_w
        self.main_metric = cfg.main_metric
        self.results: Dict = {f'BestValid/{self.main_metric}': -1., f'Test/{self.main_metric}': -1.}
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

    def train_loop(self, models: List[torch.nn.Module], train_loader: DLoader):
        for model in models:
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
            Y = batch[-1]
            # Aggregate predictions from all models
            P_agg = torch.zeros_like(Y, dtype=torch.float32).to(self.device)
            for model, loss_fn, optimizer in zip(models, self.loss_fns, self.optimizers):
                optimizer.zero_grad()
                P = model.forward(*batch[:3])
                P_agg += P / len(models)  # Averaging predictions
                loss: torch.Tensor = loss_fn(P, Y.float())
                loss.backward()
                optimizer.step()
            
            loss_val = self.loss_fns[0](P_agg, Y.float()).item()  # Use the first loss function for logging
            tqdm_bar.set_postfix(loss=f"{loss_val:.3f}")
            Y_list.append(Y)
            P_list.append(P_agg.detach())

        Y_all = torch.cat(Y_list, 0)
        P_all = torch.cat(P_list, 0)
        avg_loss = self.loss_fns[0](P_all, Y_all.float()).item()  # Use the first loss function for logging

        self.last_results = self.results
        self.results = {"Training/Loss": avg_loss}

    @torch.no_grad()
    def eval_loop(self, split, models: List[torch.nn.Module], dataloader):
        for model in models:
            model.eval()
        
        tqdm_bar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"{self.window}# Evaluating {split:5s} - {self.epoch_idx:03d}/{self.num_epoches:03d}",
            dynamic_ncols=True,
            leave=False,
        )
        Y_list = []
        P_list = []
        for idx, batch in tqdm_bar:
            batch = [v.to(self.device) for v in batch]
            Y = batch[-1]
            # Aggregate predictions from all models
            P_agg = torch.zeros_like(Y, dtype=torch.float32).to(self.device)
            for model in models:
                P = model.forward(*batch[:3])
                P_agg += P / len(models)  # Averaging predictions
            
            loss_val = self.loss_fns[0](P_agg, Y.float()).item()  # Use the first loss function for logging
            tqdm_bar.set_postfix(loss=f"{loss_val:.3f}")
            Y_list.append(Y)
            P_list.append(P_agg.detach())

        Y_all = torch.cat(Y_list, 0)
        P_all = torch.cat(P_list, 0)
        avg_loss = self.loss_fns[0](P_all, Y_all.float()).item()  # Use the first loss function for logging
        P_all, Y_all = P_all.detach().cpu(), Y_all.cpu() > 0.5
        self.results[f"{split}/Loss"] = avg_loss
        for key, metric_fn in self.metrics.items():
            P_all[torch.where(P_all.isnan())] = 0
            self.results[f"{split}/{key}"] = metric_fn(Y_all, P_all)
        pass

    def train(self, models: List[torch.nn.Module],
              train_loader: DLoader, valid_loader: DLoader,
              test_loader: DLoader):

        for epoch_idx in range(1, self.num_epoches + 1):
            self.epoch_idx = epoch_idx
            self.train_loop(models, train_loader)
            self.eval_loop("Train", models, train_loader)
            self.eval_loop("Valid", models, valid_loader)
            if self.improved():
                torch.save(models[0].state_dict(), self.log_path+'/best.pt')
                if test_loader is not None:
                    self.eval_loop("Test", models, test_loader)
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
