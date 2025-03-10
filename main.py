import hydra
from omegaconf import OmegaConf
import wandb
import warnings
from omegaconf import DictConfig

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from models.regression_module import Regression_Dense_Module


warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Initialize Wandb
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    kwargs = {
        "project": "GT-Variants-Generation",
        "name": f"{cfg.dataset.name}-{cfg.run_config.task}-{cfg.transformer_layer.type.name}",
        "config": config_dict,
        "settings": wandb.Settings(_disable_stats=True),
        "reinit": False,
        "mode": cfg.run_config.wandb_mode
    }
    wandb.init(**kwargs)

    # Load Dataset
    if cfg.dataset.name == "zinc":
        from preprocess.dataset_zinc import ZincDataset, ZincDatasetInfo
        dataset_module = ZincDataset(cfg)
        dataset_info = ZincDatasetInfo(dataset_module)
    elif cfg.dataset.name == "perov5":
        # TODO: Implement Perov5 Dataset
        pass
    else:
        raise ValueError(f"Dataset {cfg.dataset.name} not supported.")
    
    # Resuming Training
    # TODO: Implement resuming training
        
    # Initialize Model
    if cfg.run_config.task == "regression":
        if cfg.transformer_layer.dense_attention:
            model_module = Regression_Dense_Module(cfg, dataset_info)
        else:
            pass
    elif cfg.run_config.task == "generation":
        # TODO: Implement generation tasks
        pass
    else:
        raise NotImplementedError(f"Task {cfg.run_config.task} not supported.")
    
    # Callbacks
    callbacks = []
    if cfg.run_config.save_model:
        ckpt_callback = ModelCheckpoint(
            dirpath=f"checkpoints/{cfg.run_config.task}",
            filename="{epoch}",
            save_top_k=3,
            monitor="val_loss",
            mode="min",
            every_n_epochs=1
        )
        last_ckpt = ModelCheckpoint(
            dirpath=f"checkpoints/{cfg.run_config.task}",
            filename="last",
            every_n_epochs=1
        )
        callbacks.append(ckpt_callback)
        callbacks.append(last_ckpt)
    
    # Training Settings
    gpu = torch.cuda.is_available()
    trainer = Trainer(
        accelerator="gpu" if gpu else "cpu",
        devices=[cfg.run_config.gpu_id] if gpu else 1,
        max_epochs=cfg.run_config.num_epoch,
        callbacks=callbacks,
        log_every_n_steps=1,
        check_val_every_n_epoch=cfg.run_config.check_val_every_n_epoch,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        logger=[]
    )

    # Train or Test Model
    if not cfg.run_config.test_only:
        trainer.fit(model_module, dataset_module)
        trainer.test(model_module, dataset_module, verbose=False)
    else:
        trainer.test(model_module, dataset_module, ckpt_path=cfg.run_config.resume, verbose=False)


if __name__ == "__main__":
    main()
