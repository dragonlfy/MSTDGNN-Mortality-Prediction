import pickle as pkl
from typing import List, Tuple
import numpy as np
import torch as th
from torch.utils.data import DataLoader, Dataset, random_split
from Experiment.config import ExperimentConfig


class SegDataset(Dataset):
    def __init__(self, seg_list, split, window) -> None:
        super().__init__()
        self.seg_list = seg_list

        self.wind_begin_list = [seg[0] for seg in seg_list]
        categorys = [seg[1]["category"] for seg in seg_list]
        self.category_arr = th.from_numpy(np.stack(categorys))
        numerics = [seg[1]["numeric"] for seg in seg_list]
        self.numeric_arr = th.from_numpy(np.stack(numerics))

        # self.LoStay_arr = th.tensor([seg[1]['LoStay'] for seg in seg_list])
        # self.LoSurv_arr = th.tensor([seg[1]['LoSurv'] for seg in seg_list])
        ages = th.tensor([seg[1]["age"] for seg in seg_list])
        genders = th.tensor([seg[1]["gender"] for seg in seg_list])
        weights = th.tensor([seg[1]["weight"] for seg in seg_list])
        heights = th.tensor([seg[1]["height"] for seg in seg_list])
        self.demo_arr = th.stack([ages, genders, weights, heights], 1)
        mort_u_arr = th.tensor([seg[1]["mort_unit"] for seg in seg_list])
        self.labels = mort_u_arr.reshape((-1, 1)).float()

        print(
            f"{split:5s} {window} | {len(self):5d}",
            f"samples with {mort_u_arr.sum().item():4d} positives",
        )

    def __getitem__(self, index) -> th.Tensor:
        cate = self.category_arr[index]
        nume = self.numeric_arr[index]
        demo = self.demo_arr[index]
        label = self.labels[index]
        return cate, nume, demo, label

    def relalel(self, predicted, weight):
        assert 0.0 < weight < 1.0
        mask = (self.labels > 0.5) != (predicted > 0.5)
        new_labels = self.labels * (1.0 - weight) + predicted * weight
        self.labels[mask] = new_labels[mask]

    def __len__(self):
        return len(self.seg_list)


ItemType = Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]


def seg_collate_fn(item_list: List[ItemType]):
    cate_list = [item[0] for item in item_list]
    nume_list = [item[1] for item in item_list]
    demo_list = [item[2] for item in item_list]
    label_list = [item[3] for item in item_list]
    cates = th.stack(cate_list)
    numes = th.stack(nume_list)
    demos = th.stack(demo_list)
    labels = th.stack(label_list)
    return cates, numes, demos, labels


class DLoader(DataLoader):
    def __init__(self, segs_list, batch_size, shuffle, num_workers, split, window):
        seg_list = sum(segs_list, list())
        self.segdataset = SegDataset(seg_list, split, window)
        super().__init__(
            self.segdataset,
            batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=seg_collate_fn,
        )

    def relalel(self, predicted, weight):
        self.segdataset.relalel(predicted, weight)


def divide_into_groups(num_groups, segs_list):
    segs_groups = [[] for _ in range(num_groups)]
    for seg_list in segs_list:
        if seg_list is None or len(seg_list) == 0:
            continue
        for grp_idx in range(min(num_groups - 2, len(seg_list) - 1)):
            segs_groups[grp_idx].append(seg_list[grp_idx])
        for seg_idx in range(num_groups - 2, len(seg_list) - 1):
            segs_groups[-2].append(seg_list[seg_idx])
        segs_groups[-1].append(seg_list[-1])

    return segs_groups


def load_data(fpath, n_fold, i_fold, num_groups):
    with open(fpath, "rb") as rbf:
        segs_list = pkl.load(rbf)

    datasets = random_split(
        segs_list, [1 / n_fold] * n_fold, generator=th.Generator().manual_seed(42)
    )

    test_sll = [segs_list[idx] for idx in datasets.pop(i_fold).indices]
    val_sll = [segs_list[idx] for idx in datasets.pop(i_fold % (n_fold - 1)).indices]
    train_sll = [
        segs_list[idx]
        for idx in np.concatenate([dataset.indices for dataset in datasets]).tolist()
    ]

    train_segs_grp = divide_into_groups(num_groups, train_sll)
    val_segs_grp = divide_into_groups(num_groups, val_sll)
    test_segs_grp = divide_into_groups(num_groups, test_sll)
    return train_segs_grp, val_segs_grp, test_segs_grp


class MIMIC3DataModule:
    def __init__(self, cfg: ExperimentConfig) -> None:
        self.train_segs_grp, self.val_segs_grp, self.test_segs_grp = load_data(
            cfg.datapath, cfg.n_fold, cfg.i_fold, cfg.num_groups
        )
        self.cfg = cfg
        self._nextwind_loader = None

    def get_dataloaders(self, window, isolate=False):
        if window >= 0:
            w_begin = window if isolate else 0
            train_segs = self.train_segs_grp[w_begin : window + 1]
            val_segs = self.val_segs_grp[w_begin : window + 1]
            test_segs = self.test_segs_grp[w_begin : window + 1]
        elif window == -1:
            train_segs = self.train_segs_grp[window:]
            val_segs = self.val_segs_grp[window:]
            test_segs = self.test_segs_grp[window:]
            pass
        else:
            raise ValueError("Error window")

        batch_size = self.cfg.batch_size
        num_workers = self.cfg.num_workers
        if self._nextwind_loader is None:
            train_loader = DLoader(
                train_segs, batch_size, True, num_workers, "Train", window
            )
        else:
            train_loader = self._nextwind_loader
        valid_loader = DLoader(val_segs, batch_size, False, num_workers, "Val", window)
        test_loader = DLoader(test_segs, batch_size, False, num_workers, "Test", window)

        if 0 <= window < self.cfg.num_groups - 1:
            nextwind = self.train_segs_grp[window + 1 : window + 2]
            nextwind_loader = DLoader(
                nextwind, batch_size, False, num_workers, "Next", window
            )
            self._nextwind_loader = nextwind_loader
        else:
            nextwind_loader = None

        return train_loader, valid_loader, test_loader, nextwind_loader

    def get_datasets(self, window):
        loaders = self.get_dataloaders(window)
        segdatasets = [
            loader.segdataset if loader is not None else None for loader in loaders
        ]
        datasets = []

        for segdataset in segdatasets:
            if segdataset is not None:
                cate = segdataset.category_arr
                nume = segdataset.numeric_arr
                demo = segdataset.demo_arr
                demo = demo.unsqueeze(2).repeat((1, 1, 24))
                X = th.cat((cate, nume, demo), 1).numpy()
                Y = segdataset.labels.numpy()
                dataset = [X, Y]
            else:
                dataset = None
            datasets.append(dataset)

        return datasets


def test():
    from tqdm import tqdm

    cfg = ExperimentConfig("mimic3", "DPGap4SPGap48Len24Str12Art2Grp8_hf", 5, 0)
    data_module = MIMIC3DataModule(cfg)

    for window in range(cfg.num_groups):
        (
            train_loader,
            valid_loader,
            test_loader,
            nextwind_loader,
        ) = data_module.get_dataloaders(window)
        for batch in tqdm(train_loader):
            print([v.shape for v in batch])
            break

        (
            train_loader,
            valid_loader,
            test_loader,
            nextwind_loader,
        ) = data_module.get_dataloaders(0)
    nextwind_loader.relalel(nextwind_loader.dataset.labels, 0.5)

    (
        train_loader,
        valid_loader,
        test_loader,
        nextwind_loader,
    ) = data_module.get_dataloaders(1)

    for batch in tqdm(train_loader):
        print([v.shape for v in batch])
        break

    (
        train_loader,
        valid_loader,
        test_loader,
        nextwind_loader,
    ) = data_module.get_dataloaders(-1)

    for batch in tqdm(train_loader):
        print([v.shape for v in batch])
        break

    print()
