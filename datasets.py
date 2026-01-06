# from packages.MultiBench.datasets.avmnist.get_data import get_dataloader as avmnist_data_loader
from dataloader.avmnist.get_data import get_dataloader as avmnist_data_loader
# from packages.MultiBench.datasets.imdb.get_data import get_dataloader as mmimdb_data_loader
from dataloader.cub.get_data import get_dataloader as cub_data_loader
from dataloader.nyud2.get_data import get_dataloader as nyud2_data_loader
from dataloader.sunrgb_d.get_data import get_dataloader as sunrgb_d_data_loader
from dataloader.cub_mini.get_data import get_dataloader as cub_mini_data_loader
from dataloader.kinetics_2C.get_data import get_dataloader as kinetics_data_loader


DATASET_DICT = {
    "avmnist": avmnist_data_loader,
    "nyud2": nyud2_data_loader,
    "sunrgb_d": sunrgb_d_data_loader,
    "cub_mini": cub_mini_data_loader,
    "kinetics": kinetics_data_loader
}

def get_dataloader(cfg):
    dname, data_dir = cfg.experiment.dataset.name, cfg.experiment.dataset.path
    if dname not in DATASET_DICT:
        raise NotImplementedError(f"Dataset {dname} not yet supported.")
    return DATASET_DICT[dname](data_dir, batch_size=cfg.batch_size, noise_severity=cfg.noise_severity, exp_setup = cfg.exp_setup, **cfg.experiment.dataset.args)
    