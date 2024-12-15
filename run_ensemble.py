from collections import defaultdict

from Ensemble_Experiment.config import ExperimentConfig

from Experiment.save_summary import save_summary
from Experiment.seed_everything import seed_everything
from Ensemble_Experiment.ensemble_trainer import Trainer
from Ensemble_Experiment.ensemble_args import parse_args


def run(cfg: ExperimentConfig, i_fold):
    models = cfg.models
    datamodule = cfg.datamodule
    loaders = datamodule.get_dataloaders(-1)
    trainer = Trainer(f'Early_{i_fold}', cfg)
    trainer.train(models, *loaders[:3])
    return trainer.results


def main():
    args, model_args = parse_args()
    result_summary = defaultdict(list)
    for i_fold in range(args.n_fold):
        seed_everything(seed=666)
        cfg = ExperimentConfig(args.dataset, args.data, args.model, model_args, args.n_fold, i_fold)
        result = run(cfg, i_fold)
        for key, val in result.items():
            result_summary[key].append(val)

    save_summary(args.n_fold, cfg.dataset, cfg.logkey, 'Early', result_summary, model_args)


if __name__ == "__main__":
    main()
