from dataloader.avmnist.get_data import get_dataloader as avmnist_data_loader
from packages.MultiBench.datasets.imdb.get_data import get_dataloader as mmimdb_data_loader
from dataloader.nyud2.get_data import get_dataloader as nyud2_data_loader
from dataloader.clean_avmnist.get_data import get_dataloader as clean_avmnist_data_loader
from dataloader.frankfurt.get_data import get_dataloader as frankfurt_data_loader


DATASET_DICT = {
    "avmnist": avmnist_data_loader,
    "nyud2": nyud2_data_loader,
    "clean_avmnist": clean_avmnist_data_loader,
    "frankfurt": frankfurt_data_loader,
}

def get_dataloader(cfg):
    dname, data_dir = cfg.experiment.dataset.name, cfg.experiment.dataset.path
    if dname not in DATASET_DICT:
        raise NotImplementedError(f"Dataset {dname} not yet supported.")
    return DATASET_DICT[dname](data_dir, batch_size=cfg.batch_size, num_workers=cfg.num_workers, noise_severity=cfg.noise_severity, exp_setup = cfg.exp_setup, test_noise = cfg.test_noise, **cfg.experiment.dataset.args)
    