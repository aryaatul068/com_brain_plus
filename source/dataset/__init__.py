from omegaconf import DictConfig, open_dict
from .abide import load_abide_data
from .dataloader import init_dataloader, init_stratified_dataloader
from typing import List
import torch.utils as utils

def dataset_factory(cfg: DictConfig) -> List[utils.data.DataLoader]:
    assert cfg.dataset.name in ['abide']

    # Unpack 5 values: time series, graph, clinical, label, site
    final_timeseires, final_pearson, clinical_features, labels, site = eval(
        f"load_{cfg.dataset.name}_data")(cfg)

    # Choose loader and pass correct arguments
    dataloaders = init_stratified_dataloader(
        cfg, final_timeseires, final_pearson, clinical_features, labels, site
    ) if cfg.dataset.stratified else init_dataloader(
        cfg, final_timeseires, final_pearson, clinical_features, labels
    )

    return dataloaders
