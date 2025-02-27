import hydra
import logging
import warnings


warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg):
    # Logger Setup
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    file_handler = logging.FileHandler("log.txt")
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
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
    # if cfg.run_config.task == "regression":
    #     pass
    # else:
    #     raise NotImplementedError(f"Task {cfg.run_config.task} not supported.")


if __name__ == "__main__":
    main()
