from omegaconf import OmegaConf
from mGPT.config import instantiate_from_config

def build_model(cfg, datamodule):
    print("CFG.Model ",cfg.model)
    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    model_config['params']['cfg'] = cfg
    model_config['params']['datamodule'] = datamodule
    print(model_config)
    return instantiate_from_config(model_config)
