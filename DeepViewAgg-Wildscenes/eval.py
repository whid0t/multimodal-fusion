import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from torch_points3d.trainer import Trainer
import torch.multiprocessing as mp

# Set start method to spawn to allow the correct using of workers
mp.set_start_method('spawn', force=True)

@hydra.main(config_path="conf", config_name="eval")
def main(cfg):
    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly
    if cfg.pretty_print:
        print(OmegaConf.to_yaml(cfg))
    
    trainer = Trainer(cfg)
    trainer.eval()
    #
    # # https://github.com/facebookresearch/hydra/issues/440
    GlobalHydra.get_state().clear()
    return 0


if __name__ == "__main__":
    main()
