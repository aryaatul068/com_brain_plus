from .transformer import GraphTransformer
from omegaconf import DictConfig
from .brainnetcnn import BrainNetCNN
from .fbnetgen import FBNETGEN
from .COMTF import ComBrainTF
from .COMTF import ComBrainTFPlus, PromptComBrainTF

def model_factory(config: DictConfig):
    if config.model.name in ["LogisticRegression", "SVC"]:
        return None

    if config.model.name == "PromptComBrainTF":
        return PromptComBrainTF(
            node_feature_dim=config.dataset.node_feature_sz,
            time_feature_dim=config.dataset.timeseries_sz,
            meta_feature_dim=config.model.meta_feature_dim,
            hidden_dim=config.model.hidden_dim,
            num_heads=config.model.num_heads,
            num_layers=config.model.num_layers,
            prompt_len=config.model.prompt_len
        ).cpu()

    return eval(config.model.name)(config).cpu()
