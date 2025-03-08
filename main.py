import hydra
import logging
import warnings
from omegaconf import DictConfig

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from models.regression_module import Regression_Dense_Module


warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Logger Setup
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    file_handler = logging.FileHandler("log.txt")
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Load in Dataset
    if cfg.dataset.name == "zinc":
        from preprocess.dataset_zinc import ZincDataset, ZincDatasetInfo
        dataset_module = ZincDataset(cfg)
        dataset_info = ZincDatasetInfo(dataset_module)
    elif cfg.dataset.name == "perov5":
        # TODO: Implement Perov5 Dataset
        pass
    else:
        raise ValueError(f"Dataset {cfg.dataset.name} not supported.")
    logger.info(f"Dataset {cfg.dataset.name} loaded.")
    
    # Initialize Model
    if cfg.run_config.task == "regression":
        if cfg.transformer_layer.dense_attention:
            model_module = Regression_Dense_Module(cfg, logger, dataset_info)
        else:
            # TODO: Implement sparse attention
            raise NotImplementedError("Only dense attention is supported.")
    else:
        # TODO: Implement other tasks
        raise NotImplementedError(f"Task {cfg.run_config.task} not supported.")
    
    # Callbacks
    callbacks = []
    if cfg.run_config.save_model:
        ckpt_callback = ModelCheckpoint(
            dirpath=f"checkpoints/{cfg.run_config.task}-{cfg.transformer_layer.name}",
            filename="{epoch}",
            save_top_k=3,
            monitor="val_loss",
            mode="min",
            every_n_epochs=1
        )
        last_ckpt = ModelCheckpoint(
            dirpath=f"checkpoints/{cfg.run_config.task}-{cfg.transformer_layer.name}",
            filename="last",
            every_n_epochs=1
        )
        callbacks.append(ckpt_callback)
        callbacks.append(last_ckpt)
    
    # Training Settings
    gpu = torch.cuda.is_available() and cfg.run_config.use_gpu
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
        trainer.fit(model_module, dataset_module, ckpt_path=cfg.run_config.resume)
        trainer.test(model_module, dataset_module, verbose=False)
    else:
        trainer.test(model_module, dataset_module, ckpt_path=cfg.run_config.resume, verbose=False)


if __name__ == "__main__":
    main()
